#include "SeparateDeepFissionMode.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <algorithm>
#include <cctype>
#include <deque>
#include <optional>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace tulip {
namespace {
bool SeparateDeepFissionRequested = false;
}

bool isSeparateDeepFissionRequested() {
  return SeparateDeepFissionRequested;
}

void markSeparateDeepFissionRequested() {
  SeparateDeepFissionRequested = true;
}
} // namespace tulip

namespace {
static cl::opt<bool> DeepFissionDumpIR(
    "deep-fission-dump-ir",
    cl::desc("Dump deep-fission IR before and after each transformation step"),
    cl::init(false));

static cl::opt<bool> DeepFissionAllowLegacyAuthorityOnly(
    "deepfission-allow-legacy-authority-only",
    cl::desc("Allow legacy authority-only phase mitigation when carry/preload "
             "rewrite or spill fan-out is unavailable"),
    cl::init(false));

static constexpr const char *DeepFissionBuildTag = "resume-cbe-debug-v1";

struct DeepFission : public ModulePass {
  static char ID;

  struct EarlyExitGuardInfo {
    BranchInst *Branch = nullptr;
    bool TrueMeansReachedSync = true;
  };

  enum class ExitSemanticKind {
    LaneTerminating,
    PreserveActive,
  };

  struct ExitSemanticEdge {
    BranchInst *Branch = nullptr;
    unsigned SuccessorIndex = 0;
  };

  struct ExitSemanticInfo {
    ExitSemanticKind Kind = ExitSemanticKind::LaneTerminating;
    ReturnInst *RootReturn = nullptr;
    SmallVector<ExitSemanticEdge, 4> EntryEdges;
  };

  struct Stage0EntrySemanticInfo {
    BranchInst *PredicateBranch = nullptr;
    unsigned ActiveSuccessorIndex = 0;
  };

  struct BlockIndexBiasInfo {
    StringMap<int64_t> LowerThreadOffsetByBlockIdx;
  };

  struct FinalStageInfo {
    Function *WorkingStage = nullptr;
    Function *StageFunc = nullptr;
    SmallVector<GlobalVariable *, 8> UsedSpills;
    SmallVector<GlobalVariable *, 4> UsedSharedGlobals;
    SmallVector<ExitSemanticInfo, 4> ExitSemantics;
    std::optional<Stage0EntrySemanticInfo> Stage0EntrySemantics;
  };

  enum class BoundaryValueKind {
    Rematerialize,
    SpillScalar,
    SpillSimpleAllocaSeed,
    SpillReg2MemCarrier,
  };

  struct BoundaryValueInfo {
    Value *Value = nullptr;
    BoundaryValueKind Kind = BoundaryValueKind::Rematerialize;
    Type *StorageType = nullptr;
    GlobalVariable *SpillGlobal = nullptr;
    AllocaInst *CarrierAlloca = nullptr;
    Instruction *CanonicalUse = nullptr;
  };

  struct BoundaryEdgeValue {
    BasicBlock *Pred = nullptr;
    BasicBlock *OriginalSucc = nullptr;
    unsigned SuccessorIndex = 0;
    BasicBlock *EdgeBlock = nullptr;
    SmallVector<Value *, 4> ExportedValues;
  };

  struct BoundaryExitPlan {
    BasicBlock *ExitBlock = nullptr;
    SmallVector<PHINode *, 4> ExportPhis;
    SmallVector<BoundaryEdgeValue, 16> EdgeValues;
  };

  struct SemanticUndefIssue {
    Instruction *Inst = nullptr;
    std::string Reason;
  };

  enum class AuthorityAxisKind {
    None,
    ThreadIdxY,
    ThreadIdxZ,
  };

  struct CooperativePhaseRiskReport {
    bool IsBarrierFree = false;
    bool HasSharedReads = false;
    bool HasSharedWrites = false;
    bool HasCrossLaneSharedAccess = false;
    bool HasLoopContainedSharedUpdate = false;
    bool GuardedByAuthorityLane = false;
    bool HasCarryPreloadSubregion = false;
    bool HasMicroKernelRewrite = false;
    bool NeedsSpillFanout = false;
    bool HasSpillFanout = false;
    bool HasEarlyAuthorityReturnShape = false;
    bool HasRisk = false;
    std::string ResidualRiskReason;
    AuthorityAxisKind Axis = AuthorityAxisKind::None;
    uint64_t AxisModulo = 0;
    SmallVector<BasicBlock *, 8> LoopHeaders;
    SmallVector<const Instruction *, 16> SharedReads;
    SmallVector<const Instruction *, 16> SharedWrites;
    SmallVector<std::string, 8> Reasons;
  };

  struct CarrierSeedStoreIssue {
    AllocaInst *CarrierAlloca = nullptr;
    StoreInst *BadStore = nullptr;
    Instruction *NearestAccess = nullptr;
    bool ReachesLoadBeforeOverwrite = false;
  };

  std::map<Function *, std::set<CallInst *>> kernelCalls;

  DeepFission() : ModulePass(ID) { tulip::markSeparateDeepFissionRequested(); }

  void dumpFunctionIR(const std::string &Label, Function *F) {
    if (!DeepFissionDumpIR)
      return;
    errs() << "deepFission: ===== " << Label << " =====\n";
    if (!F) {
      errs() << "deepFission: <null function>\n";
      return;
    }
    errs() << "deepFission: function=" << F->getName() << "\n";
    F->print(errs());
    errs() << "\n";
  }

  void dumpInstructionIR(const std::string &Label, Instruction *I,
                         LoopInfo *LI = nullptr) {
    if (!DeepFissionDumpIR)
      return;
    errs() << "deepFission: ===== " << Label << " =====\n";
    if (!I) {
      errs() << "deepFission: <null instruction>\n";
      return;
    }
    errs() << "deepFission: instruction=" << *I << "\n";
    if (BasicBlock *BB = I->getParent()) {
      errs() << "deepFission: parent block=" << BB->getName() << "\n";
      if (Function *F = BB->getParent())
        errs() << "deepFission: parent function=" << F->getName() << "\n";
      if (LI)
        errs() << "deepFission: in loop="
               << (LI->getLoopFor(BB) ? "yes" : "no") << "\n";
    }
  }

  void dumpMarkerList(const std::string &Label,
                      const std::vector<Instruction *> &Markers,
                      LoopInfo *LI = nullptr) {
    if (!DeepFissionDumpIR)
      return;
    errs() << "deepFission: ===== " << Label << " markers=" << Markers.size()
           << " =====\n";
    for (unsigned Index = 0; Index < Markers.size(); ++Index) {
      Instruction *Marker = Markers[Index];
      errs() << "deepFission: marker[" << Index << "] ";
      if (!Marker) {
        errs() << "<null>\n";
        continue;
      }
      errs() << *Marker << "\n";
      if (BasicBlock *BB = Marker->getParent()) {
        errs() << "deepFission: marker[" << Index << "] block=" << BB->getName()
               << " function=" << BB->getParent()->getName();
        if (LI)
          errs() << " in_loop=" << (LI->getLoopFor(BB) ? "yes" : "no");
      }
      errs() << "\n";
    }
  }

  bool verifyFunctionOrReport(const std::string &Label, Function *F) {
    if (!F || F->isDeclaration())
      return true;
    if (!verifyFunction(*F, &errs()))
      return true;
    errs() << "deepFission: verifier failed after " << Label
           << " for function=" << F->getName() << "\n";
    F->print(errs());
    return false;
  }

  bool repairNonDominatingAllocaLoads(Function *F) {
    if (!F || F->isDeclaration())
      return false;

    struct DeferredRepair {
      LoadInst *Source = nullptr;
      Instruction *User = nullptr;
      unsigned OperandNo = 0;
    };

    SmallVector<DeferredRepair, 16> Repairs;
    DominatorTree DT(*F);
    for (Instruction &I : instructions(F)) {
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI)
        continue;
      auto *AI =
          dyn_cast<AllocaInst>(LI->getPointerOperand()->stripPointerCasts());
      if (!AI || AI->getFunction() != F || AI->getParent() != &F->getEntryBlock())
        continue;

      for (Use &U : LI->uses()) {
        auto *UserI = dyn_cast<Instruction>(U.getUser());
        if (!UserI || UserI == LI || isa<DbgInfoIntrinsic>(UserI))
          continue;
        if (DT.dominates(LI, UserI))
          continue;
        Repairs.push_back({LI, UserI, U.getOperandNo()});
      }
    }

    for (const DeferredRepair &Repair : Repairs) {
      auto *UserI = Repair.User;
      auto *SourceLI = Repair.Source;
      if (!UserI || !SourceLI)
        continue;

      IRBuilder<> B(UserI);
      auto *Reload =
          B.CreateLoad(SourceLI->getType(), SourceLI->getPointerOperand(),
                       SourceLI->getName() + ".repair");
      Reload->setAlignment(SourceLI->getAlignment());
      UserI->setOperand(Repair.OperandNo, Reload);
    }

    if (!Repairs.empty()) {
      errs() << "deepFission: repaired non-dominating alloca loads in "
             << F->getName() << ", count=" << Repairs.size() << "\n";
      return true;
    }
    return false;
  }

  bool isTransientSyncpointBlock(BasicBlock *BB) {
    return BB && BB->hasName() && BB->getName().startswith("syncpoint.");
  }

  bool isTriviallyCloneableResumeInst(const Instruction &I) {
    if (isa<DbgInfoIntrinsic>(I))
      return true;
    if (isa<PHINode>(I) || I.isTerminator())
      return false;
    if (auto *LI = dyn_cast<LoadInst>(&I))
      return !LI->isVolatile();
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->isVolatile())
        return false;
      return isa<AllocaInst>(SI->getPointerOperand()->stripPointerCasts());
    }
    if (auto *CI = dyn_cast<CallBase>(&I))
      return isa<DbgInfoIntrinsic>(CI);
    if (I.mayHaveSideEffects())
      return false;
    return true;
  }

  void redirectTerminatorSuccessor(BasicBlock *Pred, BasicBlock *OldSucc,
                                   BasicBlock *NewSucc) {
    if (!Pred || !OldSucc || !NewSucc)
      return;
    Instruction *TI = Pred->getTerminator();
    if (!TI)
      return;
    for (unsigned SuccIdx = 0; SuccIdx < TI->getNumSuccessors(); ++SuccIdx) {
      if (TI->getSuccessor(SuccIdx) == OldSucc)
        TI->setSuccessor(SuccIdx, NewSucc);
    }
  }

  std::string buildResumeBlockName(Function *F, BasicBlock *Target) {
    std::string Base = "df.resume";
    if (Target && Target->hasName())
      Base += "." + Target->getName().str();

    auto HasBlockNamed = [&](const std::string &Name) {
      for (BasicBlock &BB : *F) {
        if (BB.hasName() && BB.getName() == Name)
          return true;
      }
      return false;
    };

    std::string Candidate = Base;
    unsigned Suffix = 0;
    while (HasBlockNamed(Candidate))
      Candidate = Base + "." + std::to_string(++Suffix);
    return Candidate;
  }

  SmallVector<BasicBlock *, 8> collectReachableSyncpointBlocks(Function *F) {
    SmallVector<BasicBlock *, 8> Blocks;
    std::set<BasicBlock *> Reachable = collectReachableBlocksAvoiding(F, nullptr);
    for (BasicBlock *BB : Reachable) {
      if (isTransientSyncpointBlock(BB))
        Blocks.push_back(BB);
    }
    return Blocks;
  }

  bool collapseTrivialResumeForwardingBlocks(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return false;

    SmallVector<BasicBlock *, 16> Worklist;
    for (BasicBlock &BB : *F)
      Worklist.push_back(&BB);

    bool Changed = false;
    for (BasicBlock *BB : Worklist) {
      if (!BB || BB->getParent() != F || !BB->hasName())
        continue;
      if (!BB->getName().startswith("df.resume") &&
          !(isTransientSyncpointBlock(BB) &&
            BB->getName().find(".resume") != StringRef::npos))
        continue;

      auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
      if (!Br || !Br->isUnconditional())
        continue;

      BasicBlock *Succ = Br->getSuccessor(0);
      if (!Succ || Succ == BB)
        continue;

      bool Trivial = true;
      for (Instruction &I : *BB) {
        if (isa<DbgInfoIntrinsic>(&I) || I.isTerminator())
          continue;
        Trivial = false;
        break;
      }
      if (!Trivial)
        continue;

      SmallVector<BasicBlock *, 8> Preds(predecessors(BB));
      if (Preds.empty())
        continue;

      for (Instruction &I : *Succ) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break;
        int BBIdx = PN->getBasicBlockIndex(BB);
        if (BBIdx < 0)
          continue;
        Value *Incoming = PN->getIncomingValue(BBIdx);
        for (BasicBlock *Pred : Preds)
          PN->addIncoming(Incoming, Pred);
        PN->removeIncomingValue(BB, false);
      }

      for (BasicBlock *Pred : Preds)
        redirectTerminatorSuccessor(Pred, BB, Succ);

      Changed = true;
      errs() << "deepFission: [" << DeepFissionBuildTag
             << "] collapsed trivial resume forwarding block in "
             << F->getName() << " block=" << BB->getName()
             << " successor=" << Succ->getName() << "\n";
    }

    if (Changed) {
      (void)simplifyDeadControlFlow(F);
      (void)pruneUnreachableBlocks(F);
    }
    return Changed;
  }

  bool mergeSinglePredecessorSyncpointBlocks(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return false;

    bool ChangedAny = false;
    while (true) {
      bool ChangedThisRound = false;
      SmallVector<BasicBlock *, 16> Blocks;
      for (BasicBlock &BB : *F)
        Blocks.push_back(&BB);

      for (BasicBlock *BB : Blocks) {
        if (!BB || BB->getParent() != F || !isTransientSyncpointBlock(BB))
          continue;
        BasicBlock *Pred = BB->getSinglePredecessor();
        if (!Pred)
          continue;
        auto *PredBr = dyn_cast<BranchInst>(Pred->getTerminator());
        if (!PredBr || !PredBr->isUnconditional() || PredBr->getSuccessor(0) != BB)
          continue;
        std::string BlockName = BB->getName().str();
        if (!MergeBlockIntoPredecessor(BB))
          continue;
        ChangedAny = true;
        ChangedThisRound = true;
        errs() << "deepFission: [" << DeepFissionBuildTag
               << "] merged transient syncpoint block in " << F->getName()
               << " block=" << BlockName << "\n";
        break;
      }

      if (!ChangedThisRound)
        break;
    }

    return ChangedAny;
  }

  bool collectCloneableResumePrefixInsts(BasicBlock *BB, bool AllowPhis,
                                         SmallVectorImpl<Instruction *> &Insts,
                                         std::string &RejectReason) {
    if (!BB) {
      RejectReason = "null_block";
      return false;
    }

    for (Instruction &I : *BB) {
      if (isa<DbgInfoIntrinsic>(&I) || I.isTerminator())
        continue;
      if (isa<PHINode>(&I)) {
        if (AllowPhis)
          continue;
        RejectReason = "prefix_has_phi";
        return false;
      }
      if (!isTriviallyCloneableResumeInst(I)) {
        RejectReason = "noncloneable_inst";
        return false;
      }
      Insts.push_back(&I);
    }
    return true;
  }

  bool materializeCanonicalResumePrefix(Function *F, BasicBlock *Header,
                                        bool &Changed) {
    Changed = false;
    if (!F || !Header)
      return true;
    if (Header->getParent() != F || !isTransientSyncpointBlock(Header))
      return true;
    if (Header->getSinglePredecessor())
      return true;

    SmallVector<BasicBlock *, 4> HeaderPreds;
    for (BasicBlock *Pred : predecessors(Header))
      HeaderPreds.push_back(Pred);
    if (HeaderPreds.empty()) {
      errs() << "deepFission: [" << DeepFissionBuildTag
             << "] reject split-time resume materialization in "
             << F->getName() << " header=" << Header->getName()
             << " reason=no_header_preds\n";
      return true;
    }

    SmallVector<PHINode *, 8> HeaderPhis;
    for (Instruction &I : *Header) {
      auto *PN = dyn_cast<PHINode>(&I);
      if (!PN)
        break;
      HeaderPhis.push_back(PN);
    }

    SmallVector<BasicBlock *, 4> PrefixBlocks;
    SmallVector<SmallVector<Instruction *, 8>, 4> PrefixInsts;
    PrefixBlocks.push_back(Header);
    PrefixInsts.emplace_back();
    std::string RejectReason;
    if (!collectCloneableResumePrefixInsts(Header, true, PrefixInsts.back(),
                                           RejectReason)) {
      errs() << "deepFission: [" << DeepFissionBuildTag
             << "] reject split-time resume materialization in "
             << F->getName() << " header=" << Header->getName()
             << " reason=" << RejectReason << "\n";
      return true;
    }

    BasicBlock *Current = Header;
    BasicBlock *TargetBB = nullptr;
    while (true) {
      auto *CurrentBr = dyn_cast<BranchInst>(Current->getTerminator());
      if (!CurrentBr || !CurrentBr->isUnconditional()) {
        errs() << "deepFission: [" << DeepFissionBuildTag
               << "] reject split-time resume materialization in "
               << F->getName() << " header=" << Header->getName()
               << " reason=nonlinear_prefix_header block=" << Current->getName()
               << "\n";
        return true;
      }

      BasicBlock *Next = CurrentBr->getSuccessor(0);
      if (!Next || Next->getParent() != F ||
          llvm::is_contained(PrefixBlocks, Next)) {
        errs() << "deepFission: [" << DeepFissionBuildTag
               << "] reject split-time resume materialization in "
               << F->getName() << " header=" << Header->getName()
               << " reason=invalid_resume_target\n";
        return true;
      }

      auto *NextBr = dyn_cast<BranchInst>(Next->getTerminator());
      SmallVector<Instruction *, 8> NextInsts;
      std::string NextRejectReason;
      bool CanCloneNext =
          Next->getSinglePredecessor() == Current && NextBr &&
          NextBr->isUnconditional() &&
          collectCloneableResumePrefixInsts(Next, false, NextInsts,
                                            NextRejectReason);
      if (!CanCloneNext) {
        TargetBB = Next;
        break;
      }

      PrefixBlocks.push_back(Next);
      PrefixInsts.emplace_back();
      PrefixInsts.back().append(NextInsts.begin(), NextInsts.end());
      Current = Next;
    }

    LLVMContext &Context = F->getContext();
    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] split-time resume candidate function=" << F->getName()
           << " header=" << Header->getName()
           << " target=" << (TargetBB ? TargetBB->getName() : "<null>")
           << " prefix_blocks=" << PrefixBlocks.size() << "\n";
    errs() << "deepFission: [DF][SPLIT] event=resume.candidate func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " header=" << Header->getName()
           << " target=" << (TargetBB ? TargetBB->getName() : "<null>")
           << " prefix_blocks=" << PrefixBlocks.size() << " header_preds="
           << HeaderPreds.size() << " header_phis=" << HeaderPhis.size()
           << "\n";

    for (BasicBlock *Pred : HeaderPreds) {
      BasicBlock *ResumeEdge = BasicBlock::Create(
          Context, buildResumeBlockName(F, TargetBB), F, TargetBB);
      IRBuilder<> ResumeBuilder(ResumeEdge);
      ValueToValueMapTy ResumeVMap;
      for (PHINode *PN : HeaderPhis)
        ResumeVMap[PN] = PN->getIncomingValueForBlock(Pred);
      errs() << "deepFission: [DF][RESUME] event=prefix.clone.start func="
             << F->getName() << " stage=" << getStageIndexForLog(F)
             << " pred=" << Pred->getName()
             << " resume_edge=" << ResumeEdge->getName()
             << " mapped_header_phis=" << HeaderPhis.size() << "\n";

      for (unsigned BlockIdx = 0; BlockIdx < PrefixBlocks.size(); ++BlockIdx) {
        for (Instruction *I : PrefixInsts[BlockIdx]) {
          Instruction *Clone = I->clone();
          ResumeBuilder.Insert(Clone);
          ResumeVMap[I] = Clone;
          RemapInstruction(Clone, ResumeVMap,
                           RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
        }
      }
      ResumeBuilder.CreateBr(TargetBB);

      for (Instruction &I : *TargetBB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break;

        int SourceIdx = -1;
        for (auto PrefixIt = PrefixBlocks.rbegin();
             PrefixIt != PrefixBlocks.rend(); ++PrefixIt) {
          SourceIdx = PN->getBasicBlockIndex(*PrefixIt);
          if (SourceIdx >= 0)
            break;
        }
        if (SourceIdx < 0)
          continue;

        Value *Incoming = PN->getIncomingValue(SourceIdx);
        Value *Mapped = MapValue(Incoming, ResumeVMap,
                                 RF_NoModuleLevelChanges |
                                     RF_IgnoreMissingLocals);
        if (!Mapped)
          Mapped = Incoming;
        PN->addIncoming(Mapped, ResumeEdge);
        errs() << "deepFission: [DF][RESUME] event=phi.remap func="
               << F->getName() << " stage=" << getStageIndexForLog(F)
               << " target=" << TargetBB->getName() << " phi="
               << PN->getName() << " pred=" << Pred->getName()
               << " resume_edge=" << ResumeEdge->getName() << "\n";
      }

      redirectTerminatorSuccessor(Pred, Header, ResumeEdge);
      errs() << "deepFission: [DF][RESUME] event=edge.redirect func="
             << F->getName() << " stage=" << getStageIndexForLog(F)
             << " pred=" << Pred->getName() << " old_succ="
             << Header->getName() << " new_succ=" << ResumeEdge->getName()
             << "\n";
    }

    for (BasicBlock *Pred : HeaderPreds) {
      for (PHINode *PN : HeaderPhis)
        PN->removeIncomingValue(Pred, false);
    }

    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] materialized canonical resume prefix in " << F->getName()
           << " old_syncpoint=" << Header->getName()
           << " target=" << TargetBB->getName()
           << " prefix_blocks=" << PrefixBlocks.size() << "\n";
    errs() << "deepFission: [DF][SPLIT] event=resume.materialized func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " old_syncpoint=" << Header->getName() << " target="
           << TargetBB->getName() << " prefix_blocks=" << PrefixBlocks.size()
           << "\n";
    Changed = true;
    return true;
  }

  bool validateNoReachableTransientSyncpoints(Function *F, StringRef Context) {
    if (!F || F->isDeclaration() || F->empty())
      return true;

    SmallVector<BasicBlock *, 8> ReachableSyncpoints =
        collectReachableSyncpointBlocks(F);
    if (ReachableSyncpoints.empty())
      return true;

    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] reachable transient syncpoints remain after " << Context
           << " in " << F->getName() << "\n";
    for (BasicBlock *BB : ReachableSyncpoints)
      errs() << "deepFission:   block=" << BB->getName() << "\n";
    return false;
  }

  bool materializeCanonicalResumeEntries(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return true;

    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] materializeCanonicalResumeEntries scanning " << F->getName()
           << "\n";
    errs() << "deepFission: [DF][SPLIT] event=resume.scan func="
           << F->getName() << " stage=" << getStageIndexForLog(F) << "\n";
    bool ChangedAny = false;
    while (true) {
      bool ChangedThisRound = false;

      if (mergeSinglePredecessorSyncpointBlocks(F)) {
        ChangedAny = true;
        ChangedThisRound = true;
      }

      if (!ChangedThisRound) {
        SmallVector<BasicBlock *, 16> Blocks;
        for (BasicBlock &BB : *F)
          Blocks.push_back(&BB);
        for (BasicBlock *BB : Blocks) {
          if (!BB->getParent())
            continue;
          bool ChangedBlock = false;
          if (!materializeCanonicalResumePrefix(F, BB, ChangedBlock))
            return false;
          if (!ChangedBlock)
            continue;
          ChangedAny = true;
          ChangedThisRound = true;
          break;
        }
      }

      if (!ChangedThisRound)
        break;

      (void)simplifyDeadControlFlow(F);
      (void)pruneUnreachableBlocks(F);
      (void)repairNonDominatingAllocaLoads(F);
    }

    if (collectSyncMarkersInFunction(F).empty()) {
      if (!validateNoReachableTransientSyncpoints(
              F, "split-time resume materialization"))
        return false;
    } else {
      errs() << "deepFission: [" << DeepFissionBuildTag
             << "] deferred transient-syncpoint validation in " << F->getName()
             << " because reachable barriers remain\n";
    }

    if (ChangedAny)
      dumpFunctionIR("materializeCanonicalResumeEntries", F);
    else
      errs() << "deepFission: [" << DeepFissionBuildTag
             << "] materializeCanonicalResumeEntries found no changes in "
             << F->getName() << "\n";
    errs() << "deepFission: [DF][SPLIT] event=resume.scan.done func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " changed=" << (ChangedAny ? 1 : 0) << "\n";
    return true;
  }

  void collectLoopsInPostOrder(Loop *L, SmallVectorImpl<Loop *> &Loops) {
    if (!L)
      return;
    for (Loop *SubLoop : *L)
      collectLoopsInPostOrder(SubLoop, Loops);
    Loops.push_back(L);
  }

  bool loopHeaderOwnsCondition(BasicBlock *BB) {
    auto *Term = BB ? BB->getTerminator() : nullptr;
    return Term && Term->getNumSuccessors() > 1;
  }

  bool isTrivialLoopEntryForwarder(BasicBlock *BB, Loop *L,
                                   BasicBlock *&Succ) {
    Succ = nullptr;
    if (!BB || !L)
      return false;

    auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Br || !Br->isUnconditional())
      return false;

    Succ = Br->getSuccessor(0);
    if (!Succ || !L->contains(Succ))
      return false;

    for (Instruction &I : *BB) {
      if (&I == BB->getTerminator())
        continue;
      if (isa<PHINode>(&I) || isa<DbgInfoIntrinsic>(&I))
        continue;
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        Intrinsic::ID ID = II->getIntrinsicID();
        if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
          continue;
      }
      return false;
    }
    return true;
  }

  BasicBlock *resolveConditionOwningLoopEntry(BasicBlock *Header, Loop *L) {
    if (!Header || !L)
      return nullptr;

    SmallPtrSet<BasicBlock *, 8> Visited;
    BasicBlock *Current = Header;
    while (Current && Visited.insert(Current).second) {
      if (loopHeaderOwnsCondition(Current))
        return Current;

      BasicBlock *Succ = nullptr;
      if (!isTrivialLoopEntryForwarder(Current, L, Succ))
        return nullptr;
      Current = Succ;
    }
    return nullptr;
  }

  bool runBarrierFreeBasicReSSA(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return true;

    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] running barrier-free basic re-SSA cleanup in "
           << F->getName() << "\n";

    (void)promoteKernelAllocasToSSA(F);

    legacy::FunctionPassManager FPM(F->getParent());
    FPM.add(createInstructionCombiningPass());
    FPM.add(createCFGSimplificationPass());
    FPM.doInitialization();
    FPM.run(*F);
    FPM.doFinalization();

    (void)collapseTrivialResumeForwardingBlocks(F);
    (void)simplifyDeadControlFlow(F);
    (void)pruneUnreachableBlocks(F);
    (void)repairNonDominatingAllocaLoads(F);
    return true;
  }

  bool runAggressiveBarrierFreeLoopCanonicalization(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return true;

    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] running aggressive barrier-free loop canonicalization in "
           << F->getName() << "\n";

    legacy::FunctionPassManager FPM(F->getParent());
    FPM.add(createLoopSimplifyPass());
    FPM.add(createLoopRotatePass());
    FPM.add(createCFGSimplificationPass());
    FPM.doInitialization();
    FPM.run(*F);
    FPM.doFinalization();

    (void)collapseTrivialResumeForwardingBlocks(F);
    (void)simplifyDeadControlFlow(F);
    (void)pruneUnreachableBlocks(F);
    (void)repairNonDominatingAllocaLoads(F);
    return true;
  }

  bool validateBarrierFreeLoopHeaders(Function *F,
                                      bool EmitDiagnostics = true) {
    if (!F || F->isDeclaration() || F->empty())
      return true;

    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);

    SmallVector<Loop *, 16> Loops;
    for (Loop *TopLevel : LI)
      collectLoopsInPostOrder(TopLevel, Loops);

    bool Ok = true;
    for (Loop *L : Loops) {
      BasicBlock *Header = L->getHeader();
      if (!Header)
        continue;
      BasicBlock *ConditionHeader = resolveConditionOwningLoopEntry(Header, L);
      if (ConditionHeader) {
        if (EmitDiagnostics && ConditionHeader != Header) {
          errs() << "deepFission: [" << DeepFissionBuildTag
                 << "] accepted loop-entry forwarder in " << F->getName()
                 << " header=" << Header->getName()
                 << " condition_header=" << ConditionHeader->getName()
                 << "\n";
        }
        continue;
      }
      if (EmitDiagnostics) {
        errs() << "deepFission: [" << DeepFissionBuildTag
               << "] invalid barrier-free loop header in " << F->getName()
               << " header=" << Header->getName()
               << " reason=loop_entry_does_not_reach_condition_header\n";
      }
      Ok = false;
    }
    if (Ok && EmitDiagnostics) {
      errs() << "deepFission: [" << DeepFissionBuildTag
             << "] validated barrier-free loop headers in " << F->getName()
             << " loops=" << Loops.size() << "\n";
    }
    return Ok;
  }

  bool stabilizeBarrierFreeStageForCBE(Function *F,
                                       bool EnforceLoopHeaderValidation) {
    if (!F || F->isDeclaration() || F->empty())
      return true;

    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] validating barrier-free stage form in " << F->getName()
           << "\n";
    bool Changed = false;
    if (!runBarrierFreeBasicReSSA(F))
      return false;
    Changed |= collapseTrivialResumeForwardingBlocks(F);
    if (!runAggressiveBarrierFreeLoopCanonicalization(F))
      return false;
    Changed = true;
    if (!validateNoReachableTransientSyncpoints(F,
                                                "stabilized-stage validation"))
      return false;
    if (EnforceLoopHeaderValidation && !validateBarrierFreeLoopHeaders(F))
      return false;
    if (Changed)
      dumpFunctionIR("stabilizeBarrierFreeStageForCBE", F);
    return true;
  }

  bool pruneUnreachableBlocks(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return false;

    SmallPtrSet<BasicBlock *, 32> Reachable;
    SmallVector<BasicBlock *, 32> WorkList;
    BasicBlock *Entry = &F->getEntryBlock();
    Reachable.insert(Entry);
    WorkList.push_back(Entry);

    while (!WorkList.empty()) {
      BasicBlock *BB = WorkList.pop_back_val();
      for (BasicBlock *Succ : successors(BB)) {
        if (Reachable.insert(Succ).second)
          WorkList.push_back(Succ);
      }
    }

    SmallVector<BasicBlock *, 16> DeadBlocks;
    for (BasicBlock &BB : *F) {
      if (!Reachable.count(&BB))
        DeadBlocks.push_back(&BB);
    }

    if (DeadBlocks.empty())
      return false;

    DeleteDeadBlocks(DeadBlocks);
    errs() << "deepFission: pruned unreachable blocks in " << F->getName()
           << ", count=" << DeadBlocks.size() << "\n";
    return true;
  }

  bool simplifyDeadControlFlow(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return false;

    bool Changed = false;
    bool LocalChanged = false;
    do {
      LocalChanged = false;
      DominatorTree DT(*F);
      SimplifyQuery Q(F->getParent()->getDataLayout(), nullptr, &DT, nullptr);

      for (BasicBlock &BB : *F) {
        for (auto It = BB.begin(), End = BB.end(); It != End;) {
          Instruction *I = &*It++;
          if (isa<DbgInfoIntrinsic>(I) || I->isTerminator())
            continue;
          if (auto *II = dyn_cast<IntrinsicInst>(I)) {
            Intrinsic::ID ID = II->getIntrinsicID();
            if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
              continue;
          }

          if (Value *SimpleV = SimplifyInstruction(I, Q.getWithInstruction(I))) {
            if (SimpleV == I)
              continue;
            I->replaceAllUsesWith(SimpleV);
            RecursivelyDeleteTriviallyDeadInstructions(I);
            LocalChanged = true;
          }
        }
      }

      SmallVector<BasicBlock *, 16> Blocks;
      for (BasicBlock &BB : *F)
        Blocks.push_back(&BB);
      for (BasicBlock *BB : Blocks) {
        if (!BB->getParent())
          continue;
        if (ConstantFoldTerminator(BB, true, nullptr, nullptr))
          LocalChanged = true;
      }

      if (pruneUnreachableBlocks(F))
        LocalChanged = true;

      Changed |= LocalChanged;
    } while (LocalChanged);

    if (Changed)
      errs() << "deepFission: simplified constant/dead control flow in "
             << F->getName() << "\n";
    return Changed;
  }

  bool validateFinalStageSemantics(Function *F) {
    if (!F || F->isDeclaration())
      return true;
    std::string Stage = getStageIndexForLog(F);
    bool IsFinalStage =
        F->getName().contains(".df.stage") &&
        !F->getName().contains(".df.work.stage");

    SmallPtrSet<BasicBlock *, 32> Reachable;
    SmallVector<BasicBlock *, 16> WorkList;
    WorkList.push_back(&F->getEntryBlock());
    while (!WorkList.empty()) {
      BasicBlock *BB = WorkList.pop_back_val();
      if (!Reachable.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB))
        WorkList.push_back(Succ);
    }
    unsigned ReachableReturnCount = 0;
    unsigned ReachableDfExitReturnCount = 0;
    bool HasNonDfExitReachableReturn = false;
    bool DfExitRootHasNoPred = false;
    for (BasicBlock *BB : Reachable) {
      auto *RI = dyn_cast<ReturnInst>(BB->getTerminator());
      if (!RI)
        continue;
      ++ReachableReturnCount;
      if (BB->getName().startswith("df.exit")) {
        ++ReachableDfExitReturnCount;
        if (pred_empty(BB))
          DfExitRootHasNoPred = true;
      } else {
        HasNonDfExitReachableReturn = true;
      }
    }

    SmallVector<SemanticUndefIssue, 8> Issues;
    auto HasSemanticUndefOperand = [](Instruction &I) -> bool {
      for (Value *Op : I.operands()) {
        if (isa<UndefValue>(Op))
          return true;
      }
      return false;
    };

    for (BasicBlock &BB : *F) {
      if (!Reachable.count(&BB))
        continue;
      for (Instruction &I : BB) {
        if (isa<DbgInfoIntrinsic>(&I))
          continue;

        if (auto *BI = dyn_cast<BranchInst>(&I)) {
          if (BI->isConditional() && isa<UndefValue>(BI->getCondition()))
            Issues.push_back(SemanticUndefIssue{&I, "conditional branch on undef"});
          continue;
        }
        if (auto *SI = dyn_cast<SwitchInst>(&I)) {
          if (isa<UndefValue>(SI->getCondition()))
            Issues.push_back(SemanticUndefIssue{&I, "switch on undef"});
          continue;
        }
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          if (isa<UndefValue>(Store->getValueOperand()))
            Issues.push_back(SemanticUndefIssue{&I, "store of undef"});
          continue;
        }
        if (auto *PN = dyn_cast<PHINode>(&I)) {
          for (unsigned Idx = 0; Idx < PN->getNumIncomingValues(); ++Idx) {
            if (!Reachable.count(PN->getIncomingBlock(Idx)))
              continue;
            if (isa<UndefValue>(PN->getIncomingValue(Idx))) {
              Issues.push_back(
                  SemanticUndefIssue{&I, "phi with reachable undef incoming"});
              break;
            }
          }
          continue;
        }
        if ((isa<BinaryOperator>(&I) || isa<ICmpInst>(&I) ||
             isa<FCmpInst>(&I) || isa<SelectInst>(&I)) &&
            HasSemanticUndefOperand(I)) {
          Issues.push_back(
              SemanticUndefIssue{&I, "reachable value computed from undef"});
          continue;
        }
      }
    }

    if (IsFinalStage) {
      if (ReachableDfExitReturnCount == 0)
        Issues.push_back(SemanticUndefIssue{
            F->getEntryBlock().getTerminator(),
            "final stage is missing reachable df.exit return root"});
      if (HasNonDfExitReachableReturn)
        Issues.push_back(SemanticUndefIssue{
            F->getEntryBlock().getTerminator(),
            "final stage has reachable return outside df.exit root"});
      if (DfExitRootHasNoPred)
        Issues.push_back(SemanticUndefIssue{
            F->getEntryBlock().getTerminator(),
            "final stage df.exit root has no reachable predecessors"});
    }

    errs() << "deepFission: [DF][FINAL] event=validate.summary func="
           << F->getName() << " stage=" << Stage
           << " reachable_blocks=" << Reachable.size()
           << " reachable_returns=" << ReachableReturnCount
           << " reachable_df_exit_returns=" << ReachableDfExitReturnCount
           << " issues=" << Issues.size()
           << " is_final_stage=" << (IsFinalStage ? 1 : 0) << "\n";

    if (Issues.empty()) {
      errs() << "deepFission: [DF][FINAL] event=validate.ok func="
             << F->getName() << " stage=" << Stage << "\n";
      return true;
    }

    errs() << "deepFission: semantic undef validation failed for "
           << F->getName() << "\n";
    for (const SemanticUndefIssue &Issue : Issues) {
      errs() << "deepFission:   " << Issue.Reason << ": " << *Issue.Inst
             << "\n";
      errs() << "deepFission: [DF][FINAL] event=validate.issue func="
             << F->getName() << " stage=" << Stage
             << " reason=" << Issue.Reason << "\n";
    }
    return false;
  }

  std::string getDeepFissionBaseName(Function *F) {
    if (!F)
      return "df";
    std::string Name = F->getName().str();
    size_t WorkPos = Name.find(".df.work.stage");
    if (WorkPos != std::string::npos)
      Name.resize(WorkPos);
    size_t StagePos = Name.find(".df.stage");
    if (StagePos != std::string::npos)
      Name.resize(StagePos);
    size_t SplitPos = Name.find("_split");
    if (SplitPos != std::string::npos)
      Name.resize(SplitPos);
    return Name;
  }

  std::string buildWorkingStageFunctionName(Module &M, Function *F,
                                            unsigned StageIndex) {
    std::string Base = getDeepFissionBaseName(F);
    std::string Candidate =
        Base + ".df.work.stage" + std::to_string(StageIndex);
    unsigned Suffix = 0;
    while (M.getFunction(Candidate))
      Candidate = Base + ".df.work.stage" + std::to_string(StageIndex) + "." +
                  std::to_string(++Suffix);
    return Candidate;
  }

  std::string buildStageFunctionName(Module &M, Function *F, unsigned StageIndex) {
    std::string Base = getDeepFissionBaseName(F);
    std::string Candidate =
        Base + ".df.stage" + std::to_string(StageIndex);
    unsigned Suffix = 0;
    while (M.getFunction(Candidate))
      Candidate = Base + ".df.stage" + std::to_string(StageIndex) + "." +
                  std::to_string(++Suffix);
    return Candidate;
  }

  static std::optional<unsigned> getStageIndexFromFunctionName(StringRef Name) {
    auto ParseAfterToken = [&](StringRef Token) -> std::optional<unsigned> {
      size_t Pos = Name.find(Token);
      if (Pos == StringRef::npos)
        return std::nullopt;
      size_t Cursor = Pos + Token.size();
      if (Cursor >= Name.size() ||
          !std::isdigit(static_cast<unsigned char>(Name[Cursor])))
        return std::nullopt;
      unsigned Value = 0;
      while (Cursor < Name.size() &&
             std::isdigit(static_cast<unsigned char>(Name[Cursor]))) {
        Value = Value * 10 + static_cast<unsigned>(Name[Cursor] - '0');
        ++Cursor;
      }
      return Value;
    };

    if (auto Stage = ParseAfterToken(".df.work.stage"))
      return Stage;
    if (auto Stage = ParseAfterToken(".df.stage"))
      return Stage;
    return std::nullopt;
  }

  static std::string getStageIndexForLog(const Function *F) {
    if (!F)
      return "na";
    if (auto Stage = getStageIndexFromFunctionName(F->getName()))
      return std::to_string(*Stage);
    return "na";
  }

  static const char *exitSemanticKindName(ExitSemanticKind Kind) {
    switch (Kind) {
    case ExitSemanticKind::LaneTerminating:
      return "lane_terminating";
    case ExitSemanticKind::PreserveActive:
      return "preserve_active";
    }
    return "unknown";
  }

  static const char *authorityAxisKindName(AuthorityAxisKind Kind) {
    switch (Kind) {
    case AuthorityAxisKind::None:
      return "none";
    case AuthorityAxisKind::ThreadIdxY:
      return "threadIdx.y";
    case AuthorityAxisKind::ThreadIdxZ:
      return "threadIdx.z";
    }
    return "unknown";
  }

  static StringRef authorityAxisArgumentName(AuthorityAxisKind Kind) {
    switch (Kind) {
    case AuthorityAxisKind::ThreadIdxY:
      return "threadIdx.y";
    case AuthorityAxisKind::ThreadIdxZ:
      return "threadIdx.z";
    case AuthorityAxisKind::None:
      break;
    }
    return StringRef();
  }

  const char *boundaryValueKindName(BoundaryValueKind Kind) {
    switch (Kind) {
    case BoundaryValueKind::Rematerialize:
      return "rematerialize";
    case BoundaryValueKind::SpillScalar:
      return "spill-scalar";
    case BoundaryValueKind::SpillSimpleAllocaSeed:
      return "spill-simple-alloca-seed";
    case BoundaryValueKind::SpillReg2MemCarrier:
      return "spill-reg2mem-carrier";
    }
    return "unknown";
  }

  void markSyntheticReturn(ReturnInst *RI, bool PreserveActiveMask = false) {
    if (!RI)
      return;
    RI->setMetadata("tulip.df.synthetic.ret", MDNode::get(RI->getContext(), {}));
    if (PreserveActiveMask) {
      RI->setMetadata("tulip.df.synthetic.ret.preserve_active",
                      MDNode::get(RI->getContext(), {}));
    }
  }

  bool isSyntheticReturn(const ReturnInst *RI) {
    return RI && RI->getMetadata("tulip.df.synthetic.ret");
  }

  bool syntheticReturnPreservesActiveMask(const ReturnInst *RI) {
    return RI && RI->getMetadata("tulip.df.synthetic.ret.preserve_active");
  }

  void markFunctionPhaseFlag(Function *F, StringRef Key) {
    if (!F)
      return;
    F->setMetadata(Key, MDNode::get(F->getContext(), {}));
  }

  bool hasFunctionPhaseFlag(const Function *F, StringRef Key) {
    return F && F->getMetadata(Key);
  }

  void markLaneTerminatingExitBlock(BasicBlock *BB) {
    if (!BB || !BB->getTerminator())
      return;
    BB->getTerminator()->setMetadata("tulip.df.lane.terminating.exit.block",
                                     MDNode::get(BB->getContext(), {}));
  }

  bool isLaneTerminatingExitBlock(const BasicBlock *BB) {
    if (!BB)
      return false;
    if (const auto *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
      return isLaneTerminatingReturnMarked(RI);
    return BB->getTerminator() &&
           BB->getTerminator()->getMetadata(
               "tulip.df.lane.terminating.exit.block");
  }

  void markLaneTerminatingReturn(ReturnInst *RI) {
    if (!RI)
      return;
    RI->setMetadata("tulip.df.lane.terminating.ret",
                    MDNode::get(RI->getContext(), {}));
  }

  bool isLaneTerminatingReturnMarked(const ReturnInst *RI) {
    return RI && RI->getMetadata("tulip.df.lane.terminating.ret");
  }

  bool isTrivialLaneExitForwarder(const BasicBlock *BB) {
    if (!BB)
      return false;
    const auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || !BI->isUnconditional())
      return false;

    for (const Instruction &I : *BB) {
      if (&I == BI)
        continue;
      if (isa<PHINode>(I) || isa<DbgInfoIntrinsic>(I))
        continue;
      if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
        Intrinsic::ID ID = II->getIntrinsicID();
        if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
          continue;
      }
      return false;
    }
    return true;
  }

  bool isSplitStageFunction(const Function *F) {
    if (!F)
      return false;
    StringRef Name = F->getName();
    return Name.contains(".df.work.stage") || Name.contains(".df.stage");
  }

  bool blockPerformsObservableStageWork(const BasicBlock *BB) {
    if (!BB)
      return false;
    for (const Instruction &I : *BB) {
      if (I.isTerminator() || isa<PHINode>(I) || isa<DbgInfoIntrinsic>(I))
        continue;
      if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
        Intrinsic::ID ID = II->getIntrinsicID();
        if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
          continue;
      }
      if (I.mayWriteToMemory() || I.mayHaveSideEffects())
        return true;
    }
    return false;
  }

  bool returnHasObservableWorkOnIncomingRegion(const ReturnInst *RI) {
    if (!RI)
      return false;

    SmallPtrSet<const BasicBlock *, 8> ReturnRegion;
    SmallVector<const BasicBlock *, 8> Worklist;
    Worklist.push_back(RI->getParent());
    while (!Worklist.empty()) {
      const BasicBlock *BB = Worklist.pop_back_val();
      if (!ReturnRegion.insert(BB).second)
        continue;
      for (const BasicBlock *Pred : predecessors(BB)) {
        auto *BI = dyn_cast<BranchInst>(Pred->getTerminator());
        if (!BI || !BI->isUnconditional())
          continue;
        if (!isTrivialLaneExitForwarder(Pred))
          continue;
        Worklist.push_back(Pred);
      }
    }

    for (const BasicBlock *RegionBB : ReturnRegion) {
      for (const BasicBlock *Pred : predecessors(RegionBB)) {
        if (ReturnRegion.count(Pred))
          continue;
        if (blockPerformsObservableStageWork(Pred))
          return true;
      }
    }
    return false;
  }

  void collectLaneTerminatingExitPredecessors(
      BasicBlock *ExitRoot, SmallPtrSetImpl<BasicBlock *> &ExitRegion) {
    if (!ExitRoot)
      return;
    SmallVector<BasicBlock *, 8> Worklist;
    Worklist.push_back(ExitRoot);
    while (!Worklist.empty()) {
      BasicBlock *BB = Worklist.pop_back_val();
      if (!ExitRegion.insert(BB).second)
        continue;
      for (BasicBlock *Pred : predecessors(BB)) {
        auto *BI = dyn_cast<BranchInst>(Pred->getTerminator());
        if (!BI || !BI->isUnconditional())
          continue;
        if (!isTrivialLaneExitForwarder(Pred))
          continue;
        Worklist.push_back(Pred);
      }
    }
  }

  std::optional<unsigned>
  findConditionalBoundaryIntoExitRegion(BranchInst *BI,
                                        const SmallPtrSetImpl<BasicBlock *> &Region) {
    if (!BI || !BI->isConditional())
      return std::nullopt;
    std::optional<unsigned> ExitSuccIdx;
    for (unsigned I = 0; I < BI->getNumSuccessors(); ++I) {
      bool InRegion = Region.count(BI->getSuccessor(I));
      if (InRegion && !ExitSuccIdx)
        ExitSuccIdx = I;
      else if (InRegion)
        return std::nullopt;
    }
    if (!ExitSuccIdx)
      return std::nullopt;
    unsigned OtherIdx = *ExitSuccIdx ^ 1u;
    if (Region.count(BI->getSuccessor(OtherIdx)))
      return std::nullopt;
    return ExitSuccIdx;
  }

  ReturnInst *findReturnRootForExitSuccessor(BasicBlock *ExitSucc) {
    if (!ExitSucc)
      return nullptr;

    SmallPtrSet<BasicBlock *, 8> Visited;
    BasicBlock *BB = ExitSucc;
    while (BB && Visited.insert(BB).second) {
      if (auto *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
        return RI;
      auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
      if (!BI || BI->isConditional() || BI->getNumSuccessors() != 1)
        return nullptr;
      if (!isTrivialLaneExitForwarder(BB))
        return nullptr;
      BB = BI->getSuccessor(0);
    }
    
    return nullptr;
  }

  SmallVector<ExitSemanticInfo, 4>
  collectExplicitExitSemantics(Function *F,
                               bool AllowStageEntryLaneTermination = true) {
    SmallVector<ExitSemanticInfo, 4> Infos;
    if (!F || F->isDeclaration())
      return Infos;
    std::string Stage = getStageIndexForLog(F);
    errs() << "deepFission: [DF][EXIT] event=collect.start func="
           << F->getName() << " stage=" << Stage
           << " allow_stage0_lane_termination="
           << (AllowStageEntryLaneTermination ? 1 : 0) << "\n";

    if (AllowStageEntryLaneTermination) {
      if (auto Stage0Info = collectStage0EntrySemanticInfo(F)) {
        BranchInst *BI = Stage0Info->PredicateBranch;
        unsigned ExitSuccIdx = Stage0Info->ActiveSuccessorIndex ^ 1u;
        if (BI && BI->isConditional() &&
            ExitSuccIdx < BI->getNumSuccessors()) {
          BasicBlock *ExitSucc = BI->getSuccessor(ExitSuccIdx);
          if (isDriverExitSuccessor(ExitSucc)) {
            ExitSemanticInfo Info;
            Info.Kind = ExitSemanticKind::LaneTerminating;
            Info.EntryEdges.push_back({BI, ExitSuccIdx});
            if (ReturnInst *RootRI = findReturnRootForExitSuccessor(ExitSucc))
              Info.RootReturn = RootRI;
            errs() << "deepFission: explicit exit semantics collect func="
                   << F->getName() << " kind=" << exitSemanticKindName(Info.Kind)
                   << " root="
                   << (Info.RootReturn ? Info.RootReturn->getParent()->getName()
                                       : StringRef("<edge-only>"))
                   << " entry_edges=" << Info.EntryEdges.size()
                   << " source=stage0_entry\n";
            errs() << "deepFission: [DF][EXIT] event=collect.stage0_entry func="
                   << F->getName() << " stage=" << Stage
                   << " kind=" << exitSemanticKindName(Info.Kind)
                   << " root="
                   << (Info.RootReturn ? Info.RootReturn->getParent()->getName()
                                       : StringRef("<edge-only>"))
                   << " entry_edges=" << Info.EntryEdges.size() << "\n";
            errs() << "deepFission:   edge " << BI->getParent()->getName()
                   << " succ=" << ExitSuccIdx << " -> "
                   << ExitSucc->getName() << "\n";
            Infos.push_back(std::move(Info));
          }
        }
      }
    } else if (isSplitStageFunction(F)) {
      errs() << "deepFission: explicit exit semantics skipping stage0-entry "
                "lane termination for "
             << F->getName()
             << " reason=non_first_finalized_stage\n";
    }

    SmallPtrSet<ReturnInst *, 8> SeenRoots;
    for (Instruction &I : instructions(F)) {
      auto *RI = dyn_cast<ReturnInst>(&I);
      if (!RI)
        continue;

      std::optional<ExitSemanticKind> Kind;
      if (isSyntheticReturn(RI) && syntheticReturnPreservesActiveMask(RI))
        Kind = ExitSemanticKind::PreserveActive;
      else if (isLaneTerminatingReturnMarked(RI) ||
               isLaneTerminatingExitBlock(RI->getParent()))
        Kind = ExitSemanticKind::LaneTerminating;
      if (Kind && *Kind == ExitSemanticKind::LaneTerminating &&
          isSplitStageFunction(F))
        Kind = ExitSemanticKind::PreserveActive;
      if (!Kind || !SeenRoots.insert(RI).second)
        continue;
      ExitSemanticInfo Info;
      Info.Kind = *Kind;
      Info.RootReturn = RI;

      if (Info.Kind == ExitSemanticKind::LaneTerminating) {
        SmallPtrSet<BasicBlock *, 8> ExitRegion;
        collectLaneTerminatingExitPredecessors(RI->getParent(), ExitRegion);
        SmallVector<BasicBlock *, 8> RegionBlocks(ExitRegion.begin(),
                                                  ExitRegion.end());
        for (BasicBlock *RegionBB : RegionBlocks) {
          SmallVector<BasicBlock *, 8> Preds(predecessors(RegionBB));
          for (BasicBlock *PredBB : Preds) {
            if (ExitRegion.count(PredBB))
              continue;
            auto *BI = dyn_cast<BranchInst>(PredBB->getTerminator());
            auto ExitSuccIdx =
                findConditionalBoundaryIntoExitRegion(BI, ExitRegion);
            if (!ExitSuccIdx)
              continue;
            Info.EntryEdges.push_back({BI, *ExitSuccIdx});
          }
        }
      }

      errs() << "deepFission: explicit exit semantics collect func="
             << F->getName() << " kind=" << exitSemanticKindName(Info.Kind)
             << " root=" << RI->getParent()->getName()
             << " entry_edges=" << Info.EntryEdges.size() << "\n";
      errs() << "deepFission: [DF][EXIT] event=collect.root func="
             << F->getName() << " stage=" << Stage
             << " kind=" << exitSemanticKindName(Info.Kind)
             << " root=" << RI->getParent()->getName()
             << " entry_edges=" << Info.EntryEdges.size() << "\n";
      for (const ExitSemanticEdge &Edge : Info.EntryEdges) {
        errs() << "deepFission:   edge " << Edge.Branch->getParent()->getName()
               << " succ=" << Edge.SuccessorIndex << " -> "
               << Edge.Branch->getSuccessor(Edge.SuccessorIndex)->getName()
               << "\n";
      }
      Infos.push_back(std::move(Info));
    }
    errs() << "deepFission: explicit exit semantics summary func="
           << F->getName() << " count=" << Infos.size() << "\n";
    errs() << "deepFission: [DF][EXIT] event=collect.summary func="
           << F->getName() << " stage=" << Stage << " count=" << Infos.size()
           << "\n";
    return Infos;
  }

  std::optional<Stage0EntrySemanticInfo>
  collectStage0EntrySemanticInfo(Function *F) {
    if (!F || F->isDeclaration())
      return std::nullopt;

    SmallPtrSet<BasicBlock *, 8> Visited;
    BasicBlock *BB = &F->getEntryBlock();
    while (BB && Visited.insert(BB).second) {
      auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
      if (!BI)
        return std::nullopt;
      if (BI->isConditional()) {
        bool Succ0Exit = isDriverExitSuccessor(BI->getSuccessor(0));
        bool Succ1Exit = isDriverExitSuccessor(BI->getSuccessor(1));
        if (Succ0Exit == Succ1Exit)
          return std::nullopt;
        Stage0EntrySemanticInfo Info;
        Info.PredicateBranch = BI;
        Info.ActiveSuccessorIndex = Succ0Exit ? 1u : 0u;
        // Only record stage-0 entry semantics for invalidity-style guards of
        // the form `if (invalid) exit; else work;`. Positive validity guards
        // such as `if (valid) work; else exit;` appear in MG producer stages
        // and must remain preserve-active.
        if (Info.ActiveSuccessorIndex != 1u) {
          errs() << "deepFission: stage0 entry semantics skip func="
                 << F->getName() << " branch=" << BI->getParent()->getName()
                 << " reason=validity_guard active_succ="
                 << Info.ActiveSuccessorIndex << "\n";
          errs() << "deepFission: [DF][STAGE0] event=entry.skip func="
                 << F->getName() << " stage=" << getStageIndexForLog(F)
                 << " branch=" << BI->getParent()->getName()
                 << " reason=validity_guard active_succ="
                 << Info.ActiveSuccessorIndex << "\n";
          return std::nullopt;
        }
        errs() << "deepFission: stage0 entry semantics func=" << F->getName()
               << " branch=" << BI->getParent()->getName()
               << " active_succ=" << Info.ActiveSuccessorIndex << " active_bb="
               << BI->getSuccessor(Info.ActiveSuccessorIndex)->getName()
               << "\n";
        errs() << "deepFission: [DF][STAGE0] event=entry.collect func="
               << F->getName() << " stage=" << getStageIndexForLog(F)
               << " branch=" << BI->getParent()->getName()
               << " active_succ=" << Info.ActiveSuccessorIndex << " active_bb="
               << BI->getSuccessor(Info.ActiveSuccessorIndex)->getName()
               << "\n";
        return Info;
      }
      if (BI->getNumSuccessors() != 1)
        return std::nullopt;
      BB = BI->getSuccessor(0);
    }
    return std::nullopt;
  }

  bool remapExitSemantics(ArrayRef<ExitSemanticInfo> Input,
                          ValueToValueMapTy &VMap,
                          SmallVectorImpl<ExitSemanticInfo> &Output) {
    Output.clear();
    for (const ExitSemanticInfo &Info : Input) {
      ExitSemanticInfo MappedInfo;
      MappedInfo.Kind = Info.Kind;
      if (Info.RootReturn) {
        auto *MappedRoot =
            dyn_cast_or_null<ReturnInst>(VMap.lookup(Info.RootReturn));
        if (!MappedRoot)
          return false;
        MappedInfo.RootReturn = MappedRoot;
      }
      for (const ExitSemanticEdge &Edge : Info.EntryEdges) {
        auto *MappedBranch =
            dyn_cast_or_null<BranchInst>(VMap.lookup(Edge.Branch));
        if (!MappedBranch)
          return false;
        MappedInfo.EntryEdges.push_back({MappedBranch, Edge.SuccessorIndex});
      }
      errs() << "deepFission: explicit exit semantics remap kind="
             << exitSemanticKindName(MappedInfo.Kind) << " root="
             << (MappedInfo.RootReturn
                     ? MappedInfo.RootReturn->getParent()->getName()
                     : StringRef("<edge-only>"))
             << " entry_edges="
             << MappedInfo.EntryEdges.size() << "\n";
      Function *MappedFunc = nullptr;
      if (MappedInfo.RootReturn && MappedInfo.RootReturn->getParent())
        MappedFunc = MappedInfo.RootReturn->getParent()->getParent();
      else if (!MappedInfo.EntryEdges.empty() && MappedInfo.EntryEdges.front().Branch)
        MappedFunc = MappedInfo.EntryEdges.front().Branch->getFunction();
      errs() << "deepFission: [DF][EXIT] event=remap func="
             << (MappedFunc ? MappedFunc->getName() : StringRef("<null>"))
             << " stage="
             << (MappedFunc ? getStageIndexForLog(MappedFunc)
                            : std::string("na"))
             << " kind="
             << exitSemanticKindName(MappedInfo.Kind) << " root="
             << (MappedInfo.RootReturn
                     ? MappedInfo.RootReturn->getParent()->getName()
                     : StringRef("<edge-only>"))
             << " entry_edges=" << MappedInfo.EntryEdges.size() << "\n";
      Output.push_back(std::move(MappedInfo));
    }
    return true;
  }

  std::optional<Stage0EntrySemanticInfo>
  remapStage0EntrySemanticInfo(
      const std::optional<Stage0EntrySemanticInfo> &Info,
      ValueToValueMapTy &VMap) {
    if (!Info)
      return std::nullopt;
    auto *MappedBranch =
        dyn_cast_or_null<BranchInst>(VMap.lookup(Info->PredicateBranch));
    if (!MappedBranch)
      return std::nullopt;
    Stage0EntrySemanticInfo Mapped;
    Mapped.PredicateBranch = MappedBranch;
    Mapped.ActiveSuccessorIndex = Info->ActiveSuccessorIndex;
    return Mapped;
  }

  static Value *stripIntegerCasts(Value *V) {
    while (auto *CI = dyn_cast_or_null<CastInst>(V)) {
      if (!CI->getSrcTy()->isIntegerTy() || !CI->getDestTy()->isIntegerTy())
        break;
      V = CI->getOperand(0);
    }
    return V;
  }

  static bool matchLinearIndexExpr(Value *V, StringRef &BlockDimName,
                                   StringRef &BlockIdxName,
                                   StringRef &ThreadIdxName) {
    V = stripIntegerCasts(V);
    auto *BO = dyn_cast_or_null<BinaryOperator>(V);
    if (!BO || BO->getOpcode() != Instruction::Add)
      return false;

    Value *AddL = stripIntegerCasts(BO->getOperand(0));
    Value *AddR = stripIntegerCasts(BO->getOperand(1));
    auto MatchMul = [&](Value *MaybeMul, Value *MaybeTid) -> bool {
      auto *Mul = dyn_cast_or_null<BinaryOperator>(MaybeMul);
      auto *TidArg = dyn_cast_or_null<Argument>(MaybeTid);
      if (!Mul || Mul->getOpcode() != Instruction::Mul || !TidArg)
        return false;
      StringRef TidName = TidArg->getName();
      if (!TidName.startswith("threadIdx."))
        return false;
      Value *MulL = stripIntegerCasts(Mul->getOperand(0));
      Value *MulR = stripIntegerCasts(Mul->getOperand(1));
      auto *BDArg = dyn_cast_or_null<Argument>(MulL);
      auto *BIArg = dyn_cast_or_null<Argument>(MulR);
      if ((!BDArg || !BIArg) && isa<Argument>(MulR) && isa<Argument>(MulL)) {
        BDArg = cast<Argument>(MulR);
        BIArg = cast<Argument>(MulL);
      }
      if (!BDArg || !BIArg)
        return false;
      if (!BDArg->getName().startswith("blockDim.") ||
          !BIArg->getName().startswith("blockIdx."))
        return false;
      StringRef BDSuffix = BDArg->getName().substr(strlen("blockDim."));
      StringRef BISuffix = BIArg->getName().substr(strlen("blockIdx."));
      StringRef TidSuffix = TidName.substr(strlen("threadIdx."));
      if (BDSuffix != BISuffix || BISuffix != TidSuffix)
        return false;
      BlockDimName = BDArg->getName();
      BlockIdxName = BIArg->getName();
      ThreadIdxName = TidName;
      return true;
    };

    return MatchMul(AddL, AddR) || MatchMul(AddR, AddL);
  }

  static std::optional<std::pair<StringRef, int64_t>>
  tryExtractLowerThreadOffset(ICmpInst *Cmp) {
    if (!Cmp)
      return std::nullopt;

    auto TryDirectLowerBound = [&](Value *LinearV,
                                   const APInt &Bound) -> std::optional<
                                       std::pair<StringRef, int64_t>> {
      StringRef BDName, BIName, TidName;
      if (!matchLinearIndexExpr(LinearV, BDName, BIName, TidName))
        return std::nullopt;
      return std::make_pair(BIName, Bound.getSExtValue());
    };

    Value *LHS = stripIntegerCasts(Cmp->getOperand(0));
    Value *RHS = stripIntegerCasts(Cmp->getOperand(1));

    if (auto *RC = dyn_cast<ConstantInt>(RHS)) {
      switch (Cmp->getPredicate()) {
      case CmpInst::ICMP_SLT:
      case CmpInst::ICMP_ULT:
        return TryDirectLowerBound(LHS, RC->getValue());
      default:
        break;
      }
    }
    if (auto *LC = dyn_cast<ConstantInt>(LHS)) {
      switch (Cmp->getPredicate()) {
      case CmpInst::ICMP_SGT:
      case CmpInst::ICMP_UGT:
        return TryDirectLowerBound(
            RHS, LC->getValue() + APInt(LC->getBitWidth(), 1, true));
      default:
        break;
      }
    }

    auto TryShiftedLowerBound =
        [&](Value *MaybeShifted, Value *Other) -> std::optional<
            std::pair<StringRef, int64_t>> {
      auto *Shifted = dyn_cast_or_null<BinaryOperator>(stripIntegerCasts(MaybeShifted));
      auto *OtherC = dyn_cast_or_null<ConstantInt>(stripIntegerCasts(Other));
      if (!Shifted || Shifted->getOpcode() != Instruction::Add || !OtherC)
        return std::nullopt;
      auto *ShiftC =
          dyn_cast_or_null<ConstantInt>(stripIntegerCasts(Shifted->getOperand(1)));
      Value *LinearV = Shifted->getOperand(0);
      if (!ShiftC) {
        ShiftC = dyn_cast_or_null<ConstantInt>(
            stripIntegerCasts(Shifted->getOperand(0)));
        LinearV = Shifted->getOperand(1);
      }
      if (!ShiftC || ShiftC->getSExtValue() >= 0)
        return std::nullopt;
      StringRef BDName, BIName, TidName;
      if (!matchLinearIndexExpr(LinearV, BDName, BIName, TidName))
        return std::nullopt;
      int64_t LowerBound = -ShiftC->getSExtValue();
      (void)OtherC;
      return std::make_pair(BIName, LowerBound);
    };

    switch (Cmp->getPredicate()) {
    case CmpInst::ICMP_SGT:
    case CmpInst::ICMP_UGT:
    case CmpInst::ICMP_SGE:
    case CmpInst::ICMP_UGE:
      if (auto Match = TryShiftedLowerBound(LHS, RHS))
        return Match;
      break;
    default:
      break;
    }

    return std::nullopt;
  }

  static BlockIndexBiasInfo
  collectStage0BlockIndexBias(const std::optional<Stage0EntrySemanticInfo> &Info) {
    BlockIndexBiasInfo Bias;
    if (!Info || !Info->PredicateBranch)
      return Bias;
    BasicBlock *PredicateBlock = Info->PredicateBranch->getParent();
    if (!PredicateBlock)
      return Bias;
    for (Instruction &I : *PredicateBlock) {
      auto *Cmp = dyn_cast<ICmpInst>(&I);
      if (!Cmp)
        continue;
      auto Match = tryExtractLowerThreadOffset(Cmp);
      if (!Match)
        continue;
      auto It = Bias.LowerThreadOffsetByBlockIdx.find(Match->first);
      if (It == Bias.LowerThreadOffsetByBlockIdx.end())
        Bias.LowerThreadOffsetByBlockIdx[Match->first] = Match->second;
      else
        It->second = std::max(It->second, Match->second);
      errs() << "deepFission: stage0 blockidx bias candidate branch="
             << Info->PredicateBranch->getParent()->getName() << " dim="
             << Match->first << " lower_thread_offset=" << Match->second
             << "\n";
      errs() << "deepFission: [DF][STAGE0] event=bias.candidate func="
             << (Info->PredicateBranch &&
                         Info->PredicateBranch->getFunction()
                     ? Info->PredicateBranch->getFunction()->getName()
                     : StringRef("<null>"))
             << " branch=" << Info->PredicateBranch->getParent()->getName()
             << " dim=" << Match->first << " lower_thread_offset="
             << Match->second << "\n";
    }
    if (Bias.LowerThreadOffsetByBlockIdx.empty()) {
      errs() << "deepFission: stage0 blockidx bias summary branch="
             << Info->PredicateBranch->getParent()->getName()
             << " count=0\n";
      errs() << "deepFission: [DF][STAGE0] event=bias.summary func="
             << (Info->PredicateBranch &&
                         Info->PredicateBranch->getFunction()
                     ? Info->PredicateBranch->getFunction()->getName()
                     : StringRef("<null>"))
             << " branch=" << Info->PredicateBranch->getParent()->getName()
             << " count=0\n";
    } else {
      errs() << "deepFission: stage0 blockidx bias summary branch="
             << Info->PredicateBranch->getParent()->getName() << " count="
             << Bias.LowerThreadOffsetByBlockIdx.size() << "\n";
      errs() << "deepFission: [DF][STAGE0] event=bias.summary func="
             << (Info->PredicateBranch &&
                         Info->PredicateBranch->getFunction()
                     ? Info->PredicateBranch->getFunction()->getName()
                     : StringRef("<null>"))
             << " branch=" << Info->PredicateBranch->getParent()->getName()
             << " count=" << Bias.LowerThreadOffsetByBlockIdx.size() << "\n";
    }
    return Bias;
  }

  static Value *buildAdaptedBlockIndexArg(IRBuilder<> &B, CallInst *Call,
                                          Function *Callee,
                                          StringRef BlockIdxName,
                                          const BlockIndexBiasInfo *Bias) {
    if (!Call || !Callee || !Bias)
      return nullptr;
    auto It = Bias->LowerThreadOffsetByBlockIdx.find(BlockIdxName);
    if (It == Bias->LowerThreadOffsetByBlockIdx.end() || It->second <= 0)
      return nullptr;

    Value *RawBlockIdx = getCallArgForNamedParam(Call, Callee, BlockIdxName);
    if (!RawBlockIdx || !RawBlockIdx->getType()->isIntegerTy())
      return nullptr;

    StringRef Suffix = BlockIdxName.substr(strlen("blockIdx."));
    std::string BlockDimName = ("blockDim." + Suffix).str();
    Value *BlockDim = getCallArgForNamedParam(Call, Callee, BlockDimName);
    if (!BlockDim || !BlockDim->getType()->isIntegerTy())
      return nullptr;

    auto *I64Ty = Type::getInt64Ty(B.getContext());
    auto ToI64 = [&](Value *V, const Twine &Name) -> Value * {
      if (V->getType()->isIntegerTy(64))
        return V;
      return B.CreateZExtOrTrunc(V, I64Ty, Name);
    };
    Value *Raw64 = ToI64(RawBlockIdx, "df.blockidx.raw64");
    Value *BlockDim64 = ToI64(BlockDim, "df.blockdim64");
    Value *OffsetThreads =
        ConstantInt::get(I64Ty, static_cast<uint64_t>(It->second));
    Value *OffsetBlocks =
        B.CreateUDiv(OffsetThreads, BlockDim64, "df.blockidx.bias.blocks");
    Value *Adapted64 =
        B.CreateAdd(Raw64, OffsetBlocks, "df.blockidx.adapted64");
    errs() << "deepFission: adapting block index callee=" << Callee->getName()
           << " dim=" << BlockIdxName << " lower_thread_offset="
           << It->second << "\n";
    errs() << "deepFission: [DF][STAGE0] event=blockidx.adapt callee="
           << Callee->getName() << " stage=" << getStageIndexForLog(Callee)
           << " dim=" << BlockIdxName << " lower_thread_offset="
           << It->second << "\n";
    return B.CreateZExtOrTrunc(Adapted64, RawBlockIdx->getType(),
                               "df.blockidx.adapted");
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  bool hasSerializedKernelABI(Function *F) {
    if (!F || F->isDeclaration())
      return false;

    bool hasBlockDimX = false;
    bool hasBlockDimY = false;
    bool hasThreadIdxX = false;
    bool hasThreadIdxY = false;
    bool hasThreadIdxZ = false;

    for (Argument &Arg : F->args()) {
      StringRef Name = Arg.getName();
      hasBlockDimX |= (Name == "blockDim.x");
      hasBlockDimY |= (Name == "blockDim.y");
      hasThreadIdxX |= (Name == "threadIdx.x");
      hasThreadIdxY |= (Name == "threadIdx.y");
      hasThreadIdxZ |= (Name == "threadIdx.z");
    }

    return hasBlockDimX && hasBlockDimY && hasThreadIdxX && hasThreadIdxY &&
           hasThreadIdxZ;
  }

  bool isBarrierMarkerInst(Instruction *I) {
    auto *CI = dyn_cast_or_null<CallInst>(I);
    if (!CI)
      return false;
    Function *Callee = CI->getCalledFunction();
    return Callee && Callee->getName().contains("llvm.nvvm.barrier");
  }

  std::vector<Instruction *> collectSyncMarkersInFunction(Function *F) {
    std::vector<Instruction *> markers;
    if (!F || F->isDeclaration())
      return markers;

    SmallVector<BasicBlock *, 16> worklist;
    SmallPtrSet<BasicBlock *, 32> reachable;
    worklist.push_back(&F->getEntryBlock());
    while (!worklist.empty()) {
      BasicBlock *BB = worklist.pop_back_val();
      if (!reachable.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB))
        worklist.push_back(Succ);
    }

    for (BasicBlock &BB : *F) {
      if (!reachable.count(&BB))
        continue;
      for (Instruction &I : BB) {
        if (isBarrierMarkerInst(&I))
          markers.push_back(&I);
      }
    }
    return markers;
  }

  bool hasReachableSubstantiveWork(Function *F) {
    if (!F || F->isDeclaration())
      return false;

    SmallVector<BasicBlock *, 16> worklist;
    SmallPtrSet<BasicBlock *, 32> visited;
    worklist.push_back(&F->getEntryBlock());
    while (!worklist.empty()) {
      BasicBlock *BB = worklist.pop_back_val();
      if (!visited.insert(BB).second)
        continue;

      for (Instruction &I : *BB) {
        if (isa<DbgInfoIntrinsic>(&I) || I.isTerminator())
          continue;
        if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
          Intrinsic::ID ID = II->getIntrinsicID();
          if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
            continue;
        }
        if (isBarrierMarkerInst(&I))
          continue;
        return true;
      }

      for (BasicBlock *Succ : successors(BB))
        worklist.push_back(Succ);
    }
    return false;
  }

  bool hasReachableReturnBeforeSync(Function *F, Instruction *SyncInst,
                                    DominatorTree &DT) {
    if (!F || !SyncInst || SyncInst->getFunction() != F)
      return true;

    SmallVector<BasicBlock *, 16> worklist;
    SmallPtrSet<BasicBlock *, 32> reachable;
    worklist.push_back(&F->getEntryBlock());
    while (!worklist.empty()) {
      BasicBlock *BB = worklist.pop_back_val();
      if (!reachable.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB))
        worklist.push_back(Succ);
    }

    if (!reachable.count(SyncInst->getParent()))
      return true;

    for (BasicBlock &BB : *F) {
      if (!reachable.count(&BB))
        continue;
      auto *RI = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!RI)
        continue;
      if (isSyntheticReturn(RI))
        continue;
      if (!DT.dominates(SyncInst, RI))
        return true;
    }
    return false;
  }

  bool canReachBlock(BasicBlock *Start, BasicBlock *Target) {
    if (!Start || !Target)
      return false;
    if (Start == Target)
      return true;

    SmallVector<BasicBlock *, 16> worklist;
    SmallPtrSet<BasicBlock *, 32> visited;
    worklist.push_back(Start);
    while (!worklist.empty()) {
      BasicBlock *BB = worklist.pop_back_val();
      if (!visited.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB)) {
        if (Succ == Target)
          return true;
        worklist.push_back(Succ);
      }
    }
    return false;
  }

  bool canReachReturnBeforeSync(BasicBlock *Start, Instruction *SyncInst,
                                DominatorTree &DT) {
    if (!Start || !SyncInst)
      return false;

    SmallVector<BasicBlock *, 16> worklist;
    SmallPtrSet<BasicBlock *, 32> visited;
    worklist.push_back(Start);
    while (!worklist.empty()) {
      BasicBlock *BB = worklist.pop_back_val();
      if (!visited.insert(BB).second)
        continue;
      if (auto *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
        if (isSyntheticReturn(RI))
          continue;
        if (!DT.dominates(SyncInst, RI))
          return true;
      }
      for (BasicBlock *Succ : successors(BB))
        worklist.push_back(Succ);
    }
    return false;
  }

  bool collectEarlyExitGuards(Function *F, Instruction *SyncInst,
                              DominatorTree &DT,
                              SmallVectorImpl<EarlyExitGuardInfo> &Guards) {
    Guards.clear();
    if (!F || !SyncInst || !SyncInst->getParent())
      return false;

    BasicBlock *SyncBB = SyncInst->getParent();
    SmallVector<BasicBlock *, 16> worklist;
    SmallPtrSet<BasicBlock *, 32> reachable;
    worklist.push_back(&F->getEntryBlock());
    while (!worklist.empty()) {
      BasicBlock *BB = worklist.pop_back_val();
      if (!reachable.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB))
        worklist.push_back(Succ);
    }

    for (BasicBlock *BB : reachable) {
      auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
      if (!BI || !BI->isConditional())
        continue;
      if (!DT.dominates(BI, SyncInst))
        continue;

      BasicBlock *TrueBB = BI->getSuccessor(0);
      BasicBlock *FalseBB = BI->getSuccessor(1);
      bool TrueCanReachSync = canReachBlock(TrueBB, SyncBB);
      bool FalseCanReachSync = canReachBlock(FalseBB, SyncBB);
      if (TrueCanReachSync == FalseCanReachSync)
        continue;

      bool TrueCanEarlyReturn = canReachReturnBeforeSync(TrueBB, SyncInst, DT);
      bool FalseCanEarlyReturn =
          canReachReturnBeforeSync(FalseBB, SyncInst, DT);
      bool ThisTrueMeansReachedSync = false;
      bool MatchesPattern = false;
      if (TrueCanReachSync && FalseCanEarlyReturn && !FalseCanReachSync) {
        ThisTrueMeansReachedSync = true;
        MatchesPattern = true;
      } else if (FalseCanReachSync && TrueCanEarlyReturn &&
                 !TrueCanReachSync) {
        ThisTrueMeansReachedSync = false;
        MatchesPattern = true;
      }
      if (!MatchesPattern)
        continue;

      bool Seen = false;
      for (const EarlyExitGuardInfo &Info : Guards) {
        if (Info.Branch == BI) {
          Seen = true;
          break;
        }
      }
      if (!Seen)
        Guards.push_back({BI, ThisTrueMeansReachedSync});
    }
    return !Guards.empty();
  }

  std::set<BasicBlock *> collectPostSplitRegion(BasicBlock *SplitPoint) {
    std::set<BasicBlock *> Region;
    for (df_iterator<BasicBlock *> It = df_begin(SplitPoint);
         It != df_end(SplitPoint); ++It) {
      Region.insert(*It);
    }
    return Region;
  }

  std::set<BasicBlock *> collectReachableBlocksAvoiding(Function *F,
                                                        BasicBlock *Forbidden) {
    std::set<BasicBlock *> Reachable;
    if (!F || F->isDeclaration() || F->empty())
      return Reachable;

    BasicBlock *Entry = &F->getEntryBlock();
    if (Entry == Forbidden)
      return Reachable;

    SmallVector<BasicBlock *, 16> WorkList;
    WorkList.push_back(Entry);
    while (!WorkList.empty()) {
      BasicBlock *BB = WorkList.pop_back_val();
      if (!Reachable.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB)) {
        if (Succ == Forbidden)
          continue;
        WorkList.push_back(Succ);
      }
    }
    return Reachable;
  }

  std::set<BasicBlock *> collectBoundaryCutRegion(
      BasicBlock *SplitPoint, const std::set<BasicBlock *> &PostSplitRegion,
      const std::set<BasicBlock *> &EntryReachableWithoutSplit) {
    std::set<BasicBlock *> CutRegion;
    if (!SplitPoint)
      return CutRegion;

    CutRegion.insert(SplitPoint);
    for (BasicBlock *BB : PostSplitRegion) {
      if (BB == SplitPoint)
        continue;
      if (!EntryReachableWithoutSplit.count(BB))
        CutRegion.insert(BB);
    }
    return CutRegion;
  }

  std::vector<Value *> collectPostSplitLiveIns(
      Function *F, const std::set<BasicBlock *> &Region) {
    std::vector<Value *> LiveIns;
    std::set<Value *> Seen;
    for (BasicBlock &BB : *F) {
      if (!Region.count(&BB))
        continue;
      for (Instruction &I : BB) {
        for (Use &U : I.operands()) {
          Value *Op = U.get();
          if (!Op)
            continue;
          if (isa<Constant>(Op) || isa<Function>(Op) || isa<GlobalVariable>(Op))
            continue;
          if (Instruction *DefI = dyn_cast<Instruction>(Op)) {
            if (Region.count(DefI->getParent()))
              continue;
            if (Seen.insert(Op).second)
              LiveIns.push_back(Op);
          }
        }
      }
    }
    return LiveIns;
  }

  SmallVector<BasicBlock *, 8>
  collectBoundaryPreds(Function *F, const std::set<BasicBlock *> &PostSplitRegion,
                       const std::set<BasicBlock *> &EntryReachableWithoutSplit) {
    SmallVector<BasicBlock *, 8> Preds;
    SmallPtrSet<BasicBlock *, 8> Seen;
    if (!F)
      return Preds;

    for (BasicBlock &BB : *F) {
      if (!EntryReachableWithoutSplit.count(&BB))
        continue;
      auto *Term = BB.getTerminator();
      if (!Term)
        continue;
      for (unsigned SI = 0; SI < Term->getNumSuccessors(); ++SI) {
        if (!PostSplitRegion.count(Term->getSuccessor(SI)))
          continue;
        if (Seen.insert(&BB).second)
          Preds.push_back(&BB);
      }
    }
    return Preds;
  }

  SmallVector<BoundaryEdgeValue, 8>
  collectBoundaryEdges(Function *F, const std::set<BasicBlock *> &PostSplitRegion,
                       const std::set<BasicBlock *> &EntryReachableWithoutSplit) {
    SmallVector<BoundaryEdgeValue, 8> Edges;
    if (!F)
      return Edges;

    for (BasicBlock &BB : *F) {
      if (!EntryReachableWithoutSplit.count(&BB))
        continue;
      auto *Term = BB.getTerminator();
      if (!Term)
        continue;
      for (unsigned SI = 0; SI < Term->getNumSuccessors(); ++SI) {
        BasicBlock *Succ = Term->getSuccessor(SI);
        if (!PostSplitRegion.count(Succ))
          continue;
        BoundaryEdgeValue Edge;
        Edge.Pred = &BB;
        Edge.OriginalSucc = Succ;
        Edge.SuccessorIndex = SI;
        Edges.push_back(Edge);
      }
    }
    return Edges;
  }

  bool isPostRegionLocalAllocaLiveIn(Value *V,
                                     const std::set<BasicBlock *> &Region) {
    auto *AI = dyn_cast<AllocaInst>(V);
    if (!AI)
      return false;
    Function *F = AI->getFunction();
    if (!F)
      return false;
    if (AI->getParent() != &F->getEntryBlock())
      return false;

    for (User *U : AI->users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I)
        return false;
      if (Region.count(I->getParent()))
        continue;
      if (isa<DbgInfoIntrinsic>(I))
        continue;
      if (auto *II = dyn_cast<IntrinsicInst>(I)) {
        Intrinsic::ID ID = II->getIntrinsicID();
        if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
          continue;
      }
      return false;
    }
    return true;
  }

  Type *getLiveInSpillStorageType(Value *V) {
    if (!V)
      return nullptr;
    if (auto *AI = dyn_cast<AllocaInst>(V))
      return AI->getAllocatedType();
    return V->getType();
  }

  Instruction *findCanonicalBoundaryUse(Value *V, Function *F) {
    if (!V || !F)
      return nullptr;
    for (User *U : V->users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I || I->getFunction() != F || isa<DbgInfoIntrinsic>(I))
        continue;
      return I;
    }
    return dyn_cast<Instruction>(V);
  }

  bool isReg2MemCarrierAlloca(AllocaInst *AI) {
    if (!AI || !AI->getFunction())
      return false;
    if (AI->getName().contains("reg2mem"))
      return true;

    unsigned StoreCount = 0;
    bool HasStoreOutsideEntry = false;
    bool HasUndefStore = false;
    for (User *U : AI->users()) {
      auto *SI = dyn_cast<StoreInst>(U);
      if (!SI || SI->getPointerOperand()->stripPointerCasts() != AI)
        continue;
      ++StoreCount;
      HasStoreOutsideEntry |=
          SI->getParent() != &AI->getFunction()->getEntryBlock();
      HasUndefStore |= isa<UndefValue>(SI->getValueOperand());
      if (StoreCount > 1)
        break;
    }
    return HasUndefStore || HasStoreOutsideEntry || StoreCount != 1;
  }

  StoreInst *findLastStoreToAllocaBefore(AllocaInst *AI, BasicBlock *BB,
                                         Instruction *Before = nullptr) {
    if (!AI || !BB)
      return nullptr;
    BasicBlock::iterator It =
        Before ? Before->getIterator() : BB->end();
    while (It != BB->begin()) {
      --It;
      auto *SI = dyn_cast<StoreInst>(&*It);
      if (!SI || SI->getPointerOperand()->stripPointerCasts() != AI)
        continue;
      return SI;
    }
    return nullptr;
  }

  LoadInst *findLastLoadFromAllocaBefore(AllocaInst *AI, BasicBlock *BB,
                                         Instruction *Before = nullptr) {
    if (!AI || !BB)
      return nullptr;
    BasicBlock::iterator It = Before ? Before->getIterator() : BB->end();
    while (It != BB->begin()) {
      --It;
      auto *LI = dyn_cast<LoadInst>(&*It);
      if (!LI || LI->getPointerOperand()->stripPointerCasts() != AI)
        continue;
      return LI;
    }
    return nullptr;
  }

  Instruction *findLastCarrierAccessBefore(AllocaInst *AI, BasicBlock *BB,
                                           Instruction *Before = nullptr) {
    if (!AI || !BB)
      return nullptr;
    BasicBlock::iterator It = Before ? Before->getIterator() : BB->end();
    while (It != BB->begin()) {
      --It;
      Instruction *I = &*It;
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getPointerOperand()->stripPointerCasts() == AI)
          return SI;
        continue;
      }
      if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (LI->getPointerOperand()->stripPointerCasts() == AI)
          return LI;
      }
    }
    return nullptr;
  }

  bool canSpillAllocaSeedLiveIn(Value *V, Function *F, Instruction *SpillPt,
                                DominatorTree &DT) {
    auto *AI = dyn_cast<AllocaInst>(V);
    if (!AI || !F || !SpillPt || AI->getFunction() != F)
      return false;
    if (AI->getParent() != &F->getEntryBlock())
      return false;
    if (AI->isArrayAllocation())
      return false;
    Type *ElemTy = getLiveInSpillStorageType(V);
    if (!ElemTy || !ElemTy->isFirstClassType() || ElemTy->isVoidTy() ||
        ElemTy->isTokenTy() || ElemTy->isLabelTy() || ElemTy->isMetadataTy())
      return false;
    return DT.dominates(AI, SpillPt);
  }

  Value *resolveLiveInAtCallSite(Value *LiveIn, CallInst *CallI) {
    if (Argument *A = dyn_cast<Argument>(LiveIn))
      return CallI->getArgOperand(A->getArgNo());
    if (isa<Constant>(LiveIn) || isa<GlobalValue>(LiveIn))
      return LiveIn;
    return nullptr;
  }

  bool isSafeInvariantLoadForLiveIn(LoadInst *LI) {
    if (!LI || LI->isVolatile() || LI->isAtomic())
      return false;
    Value *Ptr = LI->getPointerOperand()->stripPointerCasts();
    auto *GV = dyn_cast<GlobalVariable>(Ptr);
    if (!GV)
      return false;
    if (GV->isConstant())
      return true;
    return GV->getAddressSpace() == 4;
  }

  static Argument *findNamedArgument(Function *F, StringRef Name) {
    if (!F)
      return nullptr;
    for (Argument &Arg : F->args()) {
      if (Arg.getName() == Name)
        return &Arg;
    }
    return nullptr;
  }

  Value *buildLinearThreadIndex(IRBuilder<> &B, Function *F) {
    if (!F)
      return nullptr;
    auto *I64Ty = Type::getInt64Ty(F->getContext());
    auto ToI64 = [&](Argument *Arg) -> Value * {
      if (!Arg)
        return nullptr;
      if (Arg->getType()->isIntegerTy(64))
        return Arg;
      if (!Arg->getType()->isIntegerTy())
        return nullptr;
      return B.CreateZExtOrTrunc(Arg, I64Ty, "df.tid.ext");
    };

    Value *BlockDimX = ToI64(findNamedArgument(F, "blockDim.x"));
    Value *BlockDimY = ToI64(findNamedArgument(F, "blockDim.y"));
    Value *ThreadIdxX = ToI64(findNamedArgument(F, "threadIdx.x"));
    Value *ThreadIdxY = ToI64(findNamedArgument(F, "threadIdx.y"));
    Value *ThreadIdxZ = ToI64(findNamedArgument(F, "threadIdx.z"));
    if (!BlockDimX || !BlockDimY || !ThreadIdxX || !ThreadIdxY || !ThreadIdxZ)
      return nullptr;

    Value *YZ = B.CreateAdd(
        ThreadIdxY, B.CreateMul(BlockDimY, ThreadIdxZ, "df.tid.yz.mul"),
        "df.tid.yz");
    return B.CreateAdd(ThreadIdxX,
                       B.CreateMul(BlockDimX, YZ, "df.tid.linear.mul"),
                       "df.tid.linear");
  }

  bool valueDependsOnNamedArgumentImpl(Value *V, StringRef ArgName,
                                       SmallPtrSetImpl<Value *> &Visited) {
    if (!V)
      return false;
    if (!Visited.insert(V).second)
      return false;
    if (auto *Arg = dyn_cast<Argument>(V))
      return Arg->getName() == ArgName;
    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      for (Value *Op : CE->operands()) {
        if (valueDependsOnNamedArgumentImpl(Op, ArgName, Visited))
          return true;
      }
      return false;
    }
    if (auto *I = dyn_cast<Instruction>(V)) {
      for (Value *Op : I->operands()) {
        if (valueDependsOnNamedArgumentImpl(Op, ArgName, Visited))
          return true;
      }
    }
    return false;
  }

  bool valueDependsOnNamedArgument(Value *V, StringRef ArgName) {
    SmallPtrSet<Value *, 32> Visited;
    return valueDependsOnNamedArgumentImpl(V, ArgName, Visited);
  }

  std::optional<uint64_t> findModuloDivisorFromValueImpl(
      Value *V, SmallPtrSetImpl<Value *> &Visited) {
    if (!V)
      return std::nullopt;
    if (!Visited.insert(V).second)
      return std::nullopt;

    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      if ((BO->getOpcode() == Instruction::SRem ||
           BO->getOpcode() == Instruction::URem) &&
          isa<ConstantInt>(BO->getOperand(1))) {
        uint64_t Divisor = cast<ConstantInt>(BO->getOperand(1))->getZExtValue();
        if (Divisor > 0)
          return Divisor;
      }
    }

    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      for (Value *Op : CE->operands()) {
        if (auto Divisor = findModuloDivisorFromValueImpl(Op, Visited))
          return Divisor;
      }
      return std::nullopt;
    }

    if (auto *I = dyn_cast<Instruction>(V)) {
      for (Value *Op : I->operands()) {
        if (auto Divisor = findModuloDivisorFromValueImpl(Op, Visited))
          return Divisor;
      }
    }
    return std::nullopt;
  }

  std::optional<uint64_t> findModuloDivisorFromValue(Value *V) {
    SmallPtrSet<Value *, 32> Visited;
    return findModuloDivisorFromValueImpl(V, Visited);
  }

  void collectGepIndicesFromPointerExpr(Value *Ptr,
                                        SmallVectorImpl<Value *> &Indices,
                                        SmallPtrSetImpl<Value *> &Visited) {
    if (!Ptr)
      return;
    if (!Visited.insert(Ptr).second)
      return;

    Ptr = Ptr->stripPointerCasts();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      for (unsigned I = 1; I < GEP->getNumOperands(); ++I)
        Indices.push_back(GEP->getOperand(I));
      collectGepIndicesFromPointerExpr(GEP->getPointerOperand(), Indices,
                                       Visited);
      return;
    }
    if (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        for (unsigned I = 1; I < CE->getNumOperands(); ++I)
          Indices.push_back(CE->getOperand(I));
        collectGepIndicesFromPointerExpr(CE->getOperand(0), Indices, Visited);
        return;
      }
      if (CE->getOpcode() == Instruction::BitCast ||
          CE->getOpcode() == Instruction::AddrSpaceCast) {
        collectGepIndicesFromPointerExpr(CE->getOperand(0), Indices, Visited);
        return;
      }
    }
    if (auto *PN = dyn_cast<PHINode>(Ptr)) {
      for (Value *Incoming : PN->incoming_values())
        collectGepIndicesFromPointerExpr(Incoming, Indices, Visited);
      return;
    }
    if (auto *SI = dyn_cast<SelectInst>(Ptr)) {
      collectGepIndicesFromPointerExpr(SI->getTrueValue(), Indices, Visited);
      collectGepIndicesFromPointerExpr(SI->getFalseValue(), Indices, Visited);
      return;
    }
  }

  Value *getPointerRoot(Value *Ptr) {
    if (!Ptr)
      return nullptr;
    SmallPtrSet<Value *, 16> Visited;
    Value *Current = Ptr;
    while (Current && Visited.insert(Current).second) {
      Current = Current->stripPointerCasts();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Current)) {
        Current = GEP->getPointerOperand();
        continue;
      }
      if (auto *CE = dyn_cast<ConstantExpr>(Current)) {
        if (CE->getOpcode() == Instruction::GetElementPtr ||
            CE->getOpcode() == Instruction::BitCast ||
            CE->getOpcode() == Instruction::AddrSpaceCast) {
          Current = CE->getOperand(0);
          continue;
        }
      }
      break;
    }
    return Current ? Current->stripPointerCasts() : nullptr;
  }

  bool isSharedPointerBase(Value *Base, const Function *F) {
    if (auto *Arg = dyn_cast_or_null<Argument>(Base))
      return Arg->getName().startswith("shared.");
    auto *GV = dyn_cast_or_null<GlobalVariable>(Base);
    if (!GV || !GV->isThreadLocal() || !isa<ArrayType>(GV->getValueType()))
      return false;
    if (!F)
      return true;
    std::string SpillPrefix = getDeepFissionBaseName(const_cast<Function *>(F)) +
                              ".df.spill.";
    return !GV->getName().startswith(SpillPrefix);
  }

  bool isSpillOrMaskPointerBase(Value *Base) {
    if (auto *Arg = dyn_cast_or_null<Argument>(Base)) {
      StringRef Name = Arg->getName();
      return Name.startswith("spill.") || Name == "active_mask";
    }
    return false;
  }

  bool classifySharedMemoryAccess(Instruction *I, const Function *F,
                                  bool &IsWrite, bool &IsCrossLane,
                                  AuthorityAxisKind &Axis,
                                  uint64_t &ModuloDivisor) {
    IsWrite = false;
    IsCrossLane = false;
    Axis = AuthorityAxisKind::None;
    ModuloDivisor = 0;
    if (!I)
      return false;

    Value *Ptr = nullptr;
    if (auto *LI = dyn_cast<LoadInst>(I))
      Ptr = LI->getPointerOperand();
    else if (auto *SI = dyn_cast<StoreInst>(I)) {
      Ptr = SI->getPointerOperand();
      IsWrite = true;
    } else
      return false;

    Value *Base = getPointerRoot(Ptr);
    if (!isSharedPointerBase(Base, F))
      return false;

    auto UpdateAxis = [&](AuthorityAxisKind Candidate) {
      if (Candidate == AuthorityAxisKind::None)
        return;
      if (Axis == AuthorityAxisKind::None ||
          (Axis == AuthorityAxisKind::ThreadIdxZ &&
           Candidate == AuthorityAxisKind::ThreadIdxY))
        Axis = Candidate;
    };

    SmallVector<Value *, 8> Indices;
    SmallPtrSet<Value *, 16> Visited;
    collectGepIndicesFromPointerExpr(Ptr, Indices, Visited);
    for (Value *Idx : Indices) {
      if (valueDependsOnNamedArgument(Idx, "threadIdx.y")) {
        IsCrossLane = true;
        UpdateAxis(AuthorityAxisKind::ThreadIdxY);
      } else if (valueDependsOnNamedArgument(Idx, "threadIdx.z")) {
        IsCrossLane = true;
        UpdateAxis(AuthorityAxisKind::ThreadIdxZ);
      }
      if (!ModuloDivisor) {
        if (auto Mod = findModuloDivisorFromValue(Idx))
          ModuloDivisor = *Mod;
      }
    }
    return true;
  }

  bool hasAuthorityEarlyReturnShape(const Function *F) {
    if (!F || F->isDeclaration())
      return false;
    for (const BasicBlock &BB : *F) {
      if (!BB.hasName() ||
          !BB.getName().startswith("df.phase.authority.check"))
        continue;
      auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
      if (!BI || !BI->isConditional())
        continue;
      BasicBlock *Succ0 = BI->getSuccessor(0);
      BasicBlock *Succ1 = BI->getSuccessor(1);
      auto IsSkip = [](BasicBlock *B) {
        return B && B->hasName() &&
               B->getName().startswith("df.phase.authority.skip");
      };
      auto IsBody = [](BasicBlock *B) {
        return B && B->hasName() &&
               B->getName().startswith("df.phase.authority.body");
      };
      if ((IsSkip(Succ0) && IsBody(Succ1)) ||
          (IsSkip(Succ1) && IsBody(Succ0)))
        return true;
    }
    return false;
  }

  bool detectCarryPreloadSubregion(Function *F,
                                   const CooperativePhaseRiskReport &Report) {
    if (!F || F->isDeclaration() || Report.LoopHeaders.empty())
      return false;

    errs() << "deepFission: [DF][PHASE] event=subregion.detect.start func="
           << F->getName() << " stage=" << getStageIndexForLog(F) << "\n";

    StringRef AxisArg = authorityAxisArgumentName(Report.Axis);
    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);
    SmallPtrSet<BasicBlock *, 32> RiskBlocks;
    for (BasicBlock *Header : Report.LoopHeaders) {
      if (!Header)
        continue;
      if (Loop *L = LI.getLoopFor(Header)) {
        for (BasicBlock *BB : L->blocks())
          RiskBlocks.insert(BB);
      }
    }

    unsigned PatternStores = 0;
    bool HasNonSharedLoads = false;
    for (BasicBlock *BB : RiskBlocks) {
      for (Instruction &I : *BB) {
        if (auto *LI2 = dyn_cast<LoadInst>(&I)) {
          bool IsWrite = false;
          bool IsCrossLane = false;
          AuthorityAxisKind Axis = AuthorityAxisKind::None;
          uint64_t Modulo = 0;
          if (!classifySharedMemoryAccess(LI2, F, IsWrite, IsCrossLane, Axis,
                                          Modulo)) {
            Value *Base = getPointerRoot(LI2->getPointerOperand());
            if (!isSpillOrMaskPointerBase(Base))
              HasNonSharedLoads = true;
          }
          continue;
        }

        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI)
          continue;
        bool IsWrite = false;
        bool IsCrossLane = false;
        AuthorityAxisKind Axis = AuthorityAxisKind::None;
        uint64_t Modulo = 0;
        if (!classifySharedMemoryAccess(SI, F, IsWrite, IsCrossLane, Axis,
                                        Modulo) ||
            !IsWrite)
          continue;

        if (!AxisArg.empty() &&
            !valueDependsOnNamedArgument(SI->getPointerOperand(), AxisArg))
          continue;

        auto *Sub = dyn_cast<BinaryOperator>(SI->getValueOperand());
        if (!Sub || Sub->getOpcode() != Instruction::FSub)
          continue;
        auto IsFMul = [](Value *V) {
          if (auto *BO = dyn_cast<BinaryOperator>(V))
            return BO->getOpcode() == Instruction::FMul;
          return false;
        };
        if (!IsFMul(Sub->getOperand(0)) && !IsFMul(Sub->getOperand(1)))
          continue;
        ++PatternStores;
      }
    }

    bool Found = HasNonSharedLoads && PatternStores >= 2;
    errs() << "deepFission: [DF][PHASE] event=subregion.detect.summary func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " found=" << (Found ? 1 : 0)
           << " risk_blocks=" << RiskBlocks.size()
           << " pattern_stores=" << PatternStores
           << " nonshared_loads=" << (HasNonSharedLoads ? 1 : 0) << "\n";
    return Found;
  }

  bool functionReadsSpillGlobalAlongAxis(Function *F, GlobalVariable *SpillGV,
                                         AuthorityAxisKind Axis) {
    if (!F || F->isDeclaration() || !SpillGV)
      return false;
    StringRef AxisArg = authorityAxisArgumentName(Axis);
    if (AxisArg.empty())
      return false;
    for (Instruction &I : instructions(F)) {
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI)
        continue;
      if (getPointerRoot(LI->getPointerOperand()) != SpillGV)
        continue;
      if (valueDependsOnNamedArgument(LI->getPointerOperand(), AxisArg))
        return true;
    }
    return false;
  }

  bool detectDownstreamLaneIndexedSpillUse(
      const std::vector<FinalStageInfo> &AllStages, unsigned StageIndex,
      GlobalVariable *SpillGV, AuthorityAxisKind Axis) {
    if (!SpillGV || StageIndex >= AllStages.size())
      return false;
    for (unsigned SI = StageIndex + 1; SI < AllStages.size(); ++SI) {
      Function *Consumer = AllStages[SI].WorkingStage;
      if (!Consumer)
        continue;
      bool SpillSeen = false;
      for (GlobalVariable *Used : AllStages[SI].UsedSpills) {
        if (Used == SpillGV) {
          SpillSeen = true;
          break;
        }
      }
      if (!SpillSeen)
        continue;
      if (functionReadsSpillGlobalAlongAxis(Consumer, SpillGV, Axis))
        return true;
    }
    return false;
  }

  bool valueDependsOnSharedLoadInLoop(
      Value *V, Loop *L, const Function *F, bool &CrossLaneLoadSeen,
      AuthorityAxisKind &Axis, uint64_t &ModuloDivisor,
      SmallPtrSetImpl<Value *> &Visited) {
    if (!V)
      return false;
    if (!Visited.insert(V).second)
      return false;

    if (auto *LI = dyn_cast<LoadInst>(V)) {
      bool IsWrite = false;
      bool IsCrossLane = false;
      AuthorityAxisKind LoadAxis = AuthorityAxisKind::None;
      uint64_t LoadModulo = 0;
      if (classifySharedMemoryAccess(LI, F, IsWrite, IsCrossLane, LoadAxis,
                                     LoadModulo) &&
          (!L || L->contains(LI->getParent()))) {
        if (IsCrossLane) {
          CrossLaneLoadSeen = true;
          if (Axis == AuthorityAxisKind::None ||
              (Axis == AuthorityAxisKind::ThreadIdxZ &&
               LoadAxis == AuthorityAxisKind::ThreadIdxY))
            Axis = LoadAxis;
        }
        if (!ModuloDivisor && LoadModulo)
          ModuloDivisor = LoadModulo;
        return true;
      }
    }

    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      for (Value *Op : CE->operands()) {
        if (valueDependsOnSharedLoadInLoop(Op, L, F, CrossLaneLoadSeen, Axis,
                                           ModuloDivisor, Visited))
          return true;
      }
      return false;
    }

    if (auto *I = dyn_cast<Instruction>(V)) {
      for (Value *Op : I->operands()) {
        if (valueDependsOnSharedLoadInLoop(Op, L, F, CrossLaneLoadSeen, Axis,
                                           ModuloDivisor, Visited))
          return true;
      }
    }
    return false;
  }
  
  bool isLoopHeaderPhiWithOffsetFromBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                                          Loop *L, int64_t Offset) {
    auto MatchConst = [](Value *V, int64_t &Out) -> bool {
      auto *CI = dyn_cast<ConstantInt>(V);
      if (!CI)
        return false;
      Out = CI->getSExtValue();
      return true;
    };

    int64_t C = 0;
    if (Opcode == Instruction::Add) {
      if (MatchConst(LHS, C) &&
          isLoopHeaderPhiWithOffset(RHS, L, Offset - C))
        return true;
      if (MatchConst(RHS, C) &&
          isLoopHeaderPhiWithOffset(LHS, L, Offset - C))
        return true;
    } else if (Opcode == Instruction::Sub) {
      if (MatchConst(RHS, C) &&
          isLoopHeaderPhiWithOffset(LHS, L, Offset + C))
        return true;
    }
    return false;
  }

  bool isLoopHeaderPhiWithOffset(Value *V, Loop *L, int64_t Offset) {
    if (!V || !L)
      return false;

    V = V->stripPointerCasts();
    if (auto *CI = dyn_cast<CastInst>(V))
      return isLoopHeaderPhiWithOffset(CI->getOperand(0), L, Offset);
    if (auto *PN = dyn_cast<PHINode>(V))
      return Offset == 0 && PN->getParent() == L->getHeader();

    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      return isLoopHeaderPhiWithOffsetFromBinOp(
          BO->getOpcode(), BO->getOperand(0), BO->getOperand(1), L, Offset);
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      unsigned Op = CE->getOpcode();
      if (Op == Instruction::SExt || Op == Instruction::ZExt ||
          Op == Instruction::Trunc || Op == Instruction::BitCast ||
          Op == Instruction::PtrToInt || Op == Instruction::IntToPtr)
        return isLoopHeaderPhiWithOffset(CE->getOperand(0), L, Offset);
      if (Op == Instruction::Add || Op == Instruction::Sub)
        return isLoopHeaderPhiWithOffsetFromBinOp(
            Op, CE->getOperand(0), CE->getOperand(1), L, Offset);
    }
    return false;
  }

  bool pointerExprUsesLoopHeaderPhiWithOffset(Value *Ptr, Loop *L,
                                              int64_t Offset) {
    if (!Ptr || !L)
      return false;
    SmallVector<Value *, 8> Indices;
    SmallPtrSet<Value *, 16> Visited;
    collectGepIndicesFromPointerExpr(Ptr, Indices, Visited);
    for (Value *Idx : Indices) {
      if (isLoopHeaderPhiWithOffset(Idx, L, Offset))
        return true;
    }
    return false;
  }

  bool rewriteBacksolveRecurrenceToAuthorityScalar(Function *F,
                                                   Argument *ActiveMaskArg) {
    if (!F || F->isDeclaration() || !ActiveMaskArg)
      return false;
    if (hasFunctionPhaseFlag(F, "tulip.df.phase.backsolve.recurrence.rewrite"))
      return false;

    // Guard rewrite behind full phase-risk classification so we do not
    // rewrite non-recurrence complex stages that only share partial patterns.
    CooperativePhaseRiskReport PreReport = analyzeCooperativePhaseRisk(F);
    bool HasRecurrenceReason = false;
    for (const std::string &R : PreReport.Reasons) {
      if (R == "global_backsolve_recurrence_detected") {
        HasRecurrenceReason = true;
        break;
      }
    }
    if (!PreReport.IsBarrierFree || !PreReport.HasRisk || !HasRecurrenceReason ||
        !PreReport.HasLoopContainedSharedUpdate ||
        PreReport.LoopHeaders.size() != 1 || PreReport.SharedWrites.size() != 1)
      return false;

    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);
    CooperativePhaseRiskReport ProbeReport;
    if (!detectBacksolveGlobalRecurrenceRisk(F, LI, ProbeReport))
      return false;
    if (ProbeReport.Axis != AuthorityAxisKind::ThreadIdxY ||
        ProbeReport.AxisModulo < 2)
      return false;
    const uint64_t AxisModulo = ProbeReport.AxisModulo;

    Argument *RhsArg = findNamedArgument(F, "rhs_device");
    Argument *LhsCArg = findNamedArgument(F, "lhsC_device");
    Argument *BlockDimX = findNamedArgument(F, "blockDim.x");
    Argument *BlockDimY = findNamedArgument(F, "blockDim.y");
    Argument *BlockIdxX = findNamedArgument(F, "blockIdx.x");
    Argument *BlockIdxY = findNamedArgument(F, "blockIdx.y");
    Argument *ThreadIdxX = findNamedArgument(F, "threadIdx.x");
    Argument *ThreadIdxY = findNamedArgument(F, "threadIdx.y");
    if (!RhsArg || !LhsCArg || !BlockDimX || !BlockDimY || !BlockIdxX ||
        !BlockIdxY || !ThreadIdxX || !ThreadIdxY)
      return false;

    BasicBlock *Entry = &F->getEntryBlock();
    Instruction *SplitPt = Entry->getFirstNonPHIOrDbgOrLifetime();
    if (!SplitPt || SplitPt == Entry->getTerminator())
      return false;
    BasicBlock *OldBody = Entry->splitBasicBlock(SplitPt, "df.recurrence.oldbody");
    auto *OldBr = dyn_cast<BranchInst>(Entry->getTerminator());
    if (!OldBr || !OldBr->isUnconditional())
      return false;

    LLVMContext &Ctx = F->getContext();
    Type *I1Ty = Type::getInt1Ty(Ctx);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);
    Type *F64Ty = Type::getDoubleTy(Ctx);

    auto ToI64 = [&](IRBuilder<> &B, Value *V, const Twine &Name) -> Value * {
      if (!V || !V->getType()->isIntegerTy())
        return nullptr;
      if (V->getType()->isIntegerTy(64))
        return V;
      return B.CreateZExtOrTrunc(V, I64Ty, Name);
    };

    auto BuildRhsIndex = [&](IRBuilder<> &B, Value *K64, Value *J64, Value *I64,
                             Value *Comp64) -> Value * {
      Value *Idx = B.CreateMul(K64, ConstantInt::get(I64Ty, 13), "rhs.k13");
      Idx = B.CreateAdd(Idx, J64, "rhs.kj");
      Idx = B.CreateMul(Idx, ConstantInt::get(I64Ty, 13), "rhs.kj13");
      Idx = B.CreateAdd(Idx, I64, "rhs.kji");
      Idx = B.CreateMul(Idx, ConstantInt::get(I64Ty, AxisModulo), "rhs.kji5");
      Idx = B.CreateAdd(Idx, Comp64, "rhs.idx");
      return Idx;
    };

    auto BuildLhsCIndex = [&](IRBuilder<> &B, Value *N64, Value *MM64, Value *K64,
                              Value *I64, Value *Livein5164) -> Value * {
      Value *Idx =
          B.CreateMul(N64, ConstantInt::get(I64Ty, AxisModulo), "lhsc.n5");
      Idx = B.CreateAdd(Idx, MM64, "lhsc.nm");
      Idx = B.CreateMul(Idx, ConstantInt::get(I64Ty, 12), "lhsc.nm12");
      Idx = B.CreateAdd(Idx, K64, "lhsc.k");
      Idx = B.CreateMul(Idx, ConstantInt::get(I64Ty, 13), "lhsc.k13");
      Idx = B.CreateAdd(Idx, I64, "lhsc.ki");
      Idx = B.CreateMul(Idx, ConstantInt::get(I64Ty, 11), "lhsc.ki11");
      Idx = B.CreateAdd(Idx, Livein5164, "lhsc.idx");
      return Idx;
    };

    BasicBlock *ActiveBB =
        BasicBlock::Create(Ctx, "df.recurrence.active", F, OldBody);
    BasicBlock *InactiveBB =
        BasicBlock::Create(Ctx, "df.recurrence.inactive", F, OldBody);
    BasicBlock *AuthorityCheckBB =
        BasicBlock::Create(Ctx, "df.recurrence.auth.check", F, OldBody);
    BasicBlock *AuthorityBodyBB =
        BasicBlock::Create(Ctx, "df.recurrence.auth.body", F, OldBody);
    // Final-stage semantic validator requires all reachable returns to be
    // rooted in a canonical df.exit* block.
    BasicBlock *RetBB = BasicBlock::Create(Ctx, "df.exit", F, OldBody);

    IRBuilder<> EB(OldBr);
    Value *Lane = buildLinearThreadIndex(EB, F);
    if (!Lane)
      return false;
    Value *ActiveSlot = EB.CreateInBoundsGEP(I1Ty, ActiveMaskArg, Lane,
                                             "df.recurrence.active.slot");
    Value *IsActive =
        EB.CreateLoad(I1Ty, ActiveSlot, "df.recurrence.active.load");
    EB.CreateCondBr(IsActive, ActiveBB, InactiveBB);
    OldBr->eraseFromParent();

    IRBuilder<> IB(InactiveBB);
    IB.CreateStore(ConstantInt::getFalse(Ctx), ActiveSlot);
    IB.CreateBr(RetBB);

    IRBuilder<> AB(ActiveBB);
    AB.CreateBr(AuthorityCheckBB);

    IRBuilder<> CB(AuthorityCheckBB);
    Value *MulY = CB.CreateMul(BlockDimY, BlockIdxY, "df.livein.y.mul");
    Value *Livein37 = CB.CreateAdd(MulY, ThreadIdxY, "df.livein37");
    Value *ModY = CB.CreateURem(Livein37, ConstantInt::get(I32Ty, AxisModulo),
                                "df.auth.mod");
    Value *IsAuthority =
        CB.CreateICmpEQ(ModY, ConstantInt::get(I32Ty, 0), "df.phase.authority");
    CB.CreateCondBr(IsAuthority, AuthorityBodyBB, RetBB);

    BasicBlock *ILoopHdr =
        BasicBlock::Create(Ctx, "df.recurrence.i.hdr", F, OldBody);
    BasicBlock *ILoopBody =
        BasicBlock::Create(Ctx, "df.recurrence.i.body", F, OldBody);
    BasicBlock *ILatch =
        BasicBlock::Create(Ctx, "df.recurrence.i.latch", F, OldBody);

    IRBuilder<> ABody(AuthorityBodyBB);
    Value *K32 = ABody.CreateUDiv(Livein37, ConstantInt::get(I32Ty, AxisModulo),
                                  "df.k");
    Value *MulX = ABody.CreateMul(BlockDimX, BlockIdxX, "df.livein.x.mul");
    Value *Livein51 = ABody.CreateAdd(MulX, ThreadIdxX, "df.livein51");
    Value *J32 = ABody.CreateAdd(Livein51, ConstantInt::get(I32Ty, 1), "df.j");
    Value *K64 = ToI64(ABody, K32, "df.k64");
    Value *J64 = ToI64(ABody, J32, "df.j64");
    Value *Livein5164 = ToI64(ABody, Livein51, "df.livein5164");
    if (!K64 || !J64 || !Livein5164)
      return false;
    ABody.CreateBr(ILoopHdr);

    IRBuilder<> IH(ILoopHdr);
    PHINode *IVal = IH.CreatePHI(I64Ty, 2, "df.i");
    IVal->addIncoming(ConstantInt::get(I64Ty, 9), AuthorityBodyBB);
    Value *ICond =
        IH.CreateICmpSGE(IVal, ConstantInt::getSigned(I64Ty, 0), "df.i.cond");
    IH.CreateCondBr(ICond, ILoopBody, RetBB);

    IRBuilder<> IBody(ILoopBody);
    Value *IPlus1 = IBody.CreateAdd(IVal, ConstantInt::get(I64Ty, 1), "df.i.plus1");
    for (uint64_t MM = 0; MM < AxisModulo; ++MM) {
      Value *MMVal = ConstantInt::get(I64Ty, MM);
      Value *RhsIdx = BuildRhsIndex(IBody, K64, J64, IVal, MMVal);
      Value *RhsPtr =
          IBody.CreateInBoundsGEP(F64Ty, RhsArg, RhsIdx, "df.rhs.ptr");
      Value *Acc = IBody.CreateLoad(F64Ty, RhsPtr, "df.val.init");
      for (uint64_t N = 0; N < AxisModulo; ++N) {
        Value *NVal = ConstantInt::get(I64Ty, N);
        Value *LhsIdx = BuildLhsCIndex(IBody, NVal, MMVal, K64, IVal, Livein5164);
        Value *LhsPtr =
            IBody.CreateInBoundsGEP(F64Ty, LhsCArg, LhsIdx, "df.lhsc.ptr");
        Value *LhsVal = IBody.CreateLoad(F64Ty, LhsPtr, "df.lhsc");
        Value *RhsNIdx = BuildRhsIndex(IBody, K64, J64, IPlus1, NVal);
        Value *RhsNPtr =
            IBody.CreateInBoundsGEP(F64Ty, RhsArg, RhsNIdx, "df.rhsn.ptr");
        Value *RhsNVal = IBody.CreateLoad(F64Ty, RhsNPtr, "df.rhsn");
        Value *Prod = IBody.CreateFMul(LhsVal, RhsNVal, "df.prod");
        Acc = IBody.CreateFSub(Acc, Prod, "df.acc.next");
      }
      IBody.CreateStore(Acc, RhsPtr);
    }
    IBody.CreateBr(ILatch);

    IRBuilder<> IL(ILatch);
    Value *INext = IL.CreateAdd(IVal, ConstantInt::getSigned(I64Ty, -1), "df.i.next");
    IL.CreateBr(ILoopHdr);
    IVal->addIncoming(INext, ILatch);

    IRBuilder<> RB(RetBB);
    RB.CreateRetVoid();

    markFunctionPhaseFlag(F, "tulip.df.phase.backsolve.recurrence.rewrite");
    errs() << "deepFission: [DF][PHASE] event=backsolve.recurrence.rewrite func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " applied=1\n";
    return true;
  }

  bool containsLoopHeader(SmallVectorImpl<BasicBlock *> &Headers,
                          BasicBlock *Header) {
    if (!Header)
      return false;
    for (BasicBlock *H : Headers) {
      if (H == Header)
        return true;
    }
    return false;
  }

  bool detectBacksolveGlobalRecurrenceRisk(Function *F, LoopInfo &LI,
                                           CooperativePhaseRiskReport &Report) {
    if (!F || F->isDeclaration())
      return false;

    Argument *RhsArg = findNamedArgument(F, "rhs_device");
    if (!RhsArg)
      return false;

    bool Found = false;
    SmallVector<Loop *, 8> Worklist;
    for (Loop *Top : LI)
      Worklist.push_back(Top);

    for (unsigned I = 0; I < Worklist.size(); ++I) {
      Loop *L = Worklist[I];
      if (!L)
        continue;
      for (Loop *Sub : *L)
        Worklist.push_back(Sub);

      bool HasStoreAtI = false;
      bool HasLoadAtIPlus1 = false;
      bool HasLaneAxisUse = false;
      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &Inst : *BB) {
          if (auto *SI = dyn_cast<StoreInst>(&Inst)) {
            if (getPointerRoot(SI->getPointerOperand()) != RhsArg)
              continue;
            if (pointerExprUsesLoopHeaderPhiWithOffset(SI->getPointerOperand(), L,
                                                       0))
              HasStoreAtI = true;
            if (valueDependsOnNamedArgument(SI->getPointerOperand(),
                                            "threadIdx.y"))
              HasLaneAxisUse = true;
            continue;
          }
          auto *LI2 = dyn_cast<LoadInst>(&Inst);
          if (!LI2)
            continue;
          if (getPointerRoot(LI2->getPointerOperand()) != RhsArg)
            continue;
          if (pointerExprUsesLoopHeaderPhiWithOffset(LI2->getPointerOperand(), L,
                                                     1))
            HasLoadAtIPlus1 = true;
          if (valueDependsOnNamedArgument(LI2->getPointerOperand(),
                                          "threadIdx.y"))
            HasLaneAxisUse = true;
        }
      }

      if (!HasStoreAtI || !HasLoadAtIPlus1)
        continue;

      Found = true;
      if (!containsLoopHeader(Report.LoopHeaders, L->getHeader()))
        Report.LoopHeaders.push_back(L->getHeader());
      Report.HasLoopContainedSharedUpdate = true;
      Report.HasCrossLaneSharedAccess |= HasLaneAxisUse;
      if (Report.Axis == AuthorityAxisKind::None ||
          Report.Axis == AuthorityAxisKind::ThreadIdxZ)
        Report.Axis = AuthorityAxisKind::ThreadIdxY;
      if (!Report.AxisModulo)
        Report.AxisModulo = 5;
    }

    if (Found) {
      errs() << "deepFission: [DF][PHASE] event=detect.backsolve_recurrence func="
             << F->getName() << " stage=" << getStageIndexForLog(F)
             << " axis=" << authorityAxisKindName(Report.Axis)
             << " axis_modulo=" << Report.AxisModulo << "\n";
    }
    return Found;
  }

  BasicBlock *findActiveBodyEntryBlock(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return nullptr;
    BasicBlock *EntryBB = &F->getEntryBlock();
    auto *BI = dyn_cast<BranchInst>(EntryBB->getTerminator());
    if (!BI)
      return nullptr;
    if (BI->isConditional()) {
      BasicBlock *Succ0 = BI->getSuccessor(0);
      BasicBlock *Succ1 = BI->getSuccessor(1);
      auto IsSkipBlock = [](BasicBlock *BB) {
        return BB && BB->hasName() &&
               BB->getName().startswith("df.active.skip");
      };
      if (IsSkipBlock(Succ0) && !IsSkipBlock(Succ1))
        return Succ1;
      if (IsSkipBlock(Succ1) && !IsSkipBlock(Succ0))
        return Succ0;
      return nullptr;
    }
    if (BI->isUnconditional())
      return BI->getSuccessor(0);
    return nullptr;
  }

  CooperativePhaseRiskReport analyzeCooperativePhaseRisk(Function *F) {
    CooperativePhaseRiskReport Report;
    if (!F || F->isDeclaration()) {
      Report.Reasons.push_back("invalid_function");
      return Report;
    }

    std::string Stage = getStageIndexForLog(F);
    errs() << "deepFission: [DF][PHASE] event=detect.start func=" << F->getName()
           << " stage=" << Stage << "\n";

    Report.IsBarrierFree = collectSyncMarkersInFunction(F).empty();

    DominatorTree DT(*F);
    LoopInfo LI;
    LI.analyze(DT);

    struct LoopPhaseStats {
      BasicBlock *Header = nullptr;
      unsigned SharedReads = 0;
      unsigned SharedWrites = 0;
      bool HasCrossLane = false;
      bool HasStoreFromSharedLoad = false;
      bool HasNonSharedStore = false;
    };

    DenseMap<Loop *, LoopPhaseStats> LoopStats;
    auto UpdateAxis = [&](AuthorityAxisKind Candidate) {
      if (Candidate == AuthorityAxisKind::None)
        return;
      if (Report.Axis == AuthorityAxisKind::None ||
          (Report.Axis == AuthorityAxisKind::ThreadIdxZ &&
           Candidate == AuthorityAxisKind::ThreadIdxY))
        Report.Axis = Candidate;
    };

    for (Instruction &I : instructions(F)) {
      bool IsWrite = false;
      bool IsCrossLane = false;
      AuthorityAxisKind AccessAxis = AuthorityAxisKind::None;
      uint64_t AccessModulo = 0;
      bool IsShared =
          classifySharedMemoryAccess(&I, F, IsWrite, IsCrossLane, AccessAxis,
                                     AccessModulo);

      Loop *L = LI.getLoopFor(I.getParent());
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!IsShared && L) {
          Value *StoreBase = getPointerRoot(SI->getPointerOperand());
          if (!isSpillOrMaskPointerBase(StoreBase))
            LoopStats[L].HasNonSharedStore = true;
        }
      }

      if (!IsShared)
        continue;

      if (IsWrite) {
        Report.HasSharedWrites = true;
        Report.SharedWrites.push_back(&I);
      } else {
        Report.HasSharedReads = true;
        Report.SharedReads.push_back(&I);
      }
      if (IsCrossLane)
        Report.HasCrossLaneSharedAccess = true;
      UpdateAxis(AccessAxis);
      if (!Report.AxisModulo && AccessModulo)
        Report.AxisModulo = AccessModulo;

      if (!L)
        continue;
      LoopPhaseStats &Stats = LoopStats[L];
      if (!Stats.Header)
        Stats.Header = L->getHeader();
      if (IsWrite)
        ++Stats.SharedWrites;
      else
        ++Stats.SharedReads;
      if (IsCrossLane)
        Stats.HasCrossLane = true;

      if (!IsWrite)
        continue;

      auto *SI = cast<StoreInst>(&I);
      bool CrossLaneLoadSeen = false;
      AuthorityAxisKind ValueAxis = AuthorityAxisKind::None;
      uint64_t ValueModulo = 0;
      SmallPtrSet<Value *, 32> Visited;
      if (valueDependsOnSharedLoadInLoop(SI->getValueOperand(), L, F,
                                         CrossLaneLoadSeen, ValueAxis,
                                         ValueModulo, Visited)) {
        Stats.HasStoreFromSharedLoad = true;
      }
      if (CrossLaneLoadSeen)
        Stats.HasCrossLane = true;
      UpdateAxis(ValueAxis);
      if (!Report.AxisModulo && ValueModulo)
        Report.AxisModulo = ValueModulo;
    }

    for (const auto &KV : LoopStats) {
      const LoopPhaseStats &Stats = KV.second;
      if (!Stats.Header)
        continue;
      if (Stats.SharedReads == 0 || Stats.SharedWrites == 0)
        continue;
      if (!Stats.HasCrossLane || !Stats.HasStoreFromSharedLoad)
        continue;
      if (!Stats.HasNonSharedStore)
        continue;
      Report.HasLoopContainedSharedUpdate = true;
      Report.LoopHeaders.push_back(Stats.Header);
    }

    bool HasBacksolveGlobalRecurrence =
        detectBacksolveGlobalRecurrenceRisk(F, LI, Report);
    bool RawRisk = (Report.IsBarrierFree && Report.HasCrossLaneSharedAccess &&
                    Report.HasLoopContainedSharedUpdate) ||
                   (Report.IsBarrierFree && HasBacksolveGlobalRecurrence);
    Report.HasCarryPreloadSubregion = detectCarryPreloadSubregion(F, Report);
    Report.HasEarlyAuthorityReturnShape =
        hasAuthorityEarlyReturnShape(F) ||
        hasFunctionPhaseFlag(F, "tulip.df.phase.authority.early_return");
    Report.HasMicroKernelRewrite =
        hasFunctionPhaseFlag(F, "tulip.df.phase.microkernel.rewrite");
    Report.NeedsSpillFanout =
        hasFunctionPhaseFlag(F, "tulip.df.phase.spill.fanout.required");
    Report.HasSpillFanout =
        hasFunctionPhaseFlag(F, "tulip.df.phase.spill.fanout");

    if (RawRisk) {
      auto IsAuthoritySkipExit = [this](BasicBlock *BB) -> bool {
        if (!BB)
          return false;
        Instruction *TI = BB->getTerminator();
        if (auto *RI = dyn_cast<ReturnInst>(TI))
          return isSyntheticReturn(RI) && syntheticReturnPreservesActiveMask(RI);
        auto *BI = dyn_cast<BranchInst>(TI);
        if (!BI || !BI->isUnconditional())
          return false;
        BasicBlock *Succ = BI->getSuccessor(0);
        return Succ && Succ->hasName() && Succ->getName().startswith("df.exit");
      };
      for (BasicBlock &BB : *F) {
        if (!BB.hasName() || !BB.getName().startswith("df.phase.authority.skip"))
          continue;
        if (!IsAuthoritySkipExit(&BB))
          continue;
        for (BasicBlock *Pred : predecessors(&BB)) {
          auto *BI = dyn_cast<BranchInst>(Pred->getTerminator());
          if (!BI || !BI->isConditional())
            continue;
          BasicBlock *AuthorityBody =
              BI->getSuccessor(0) == &BB ? BI->getSuccessor(1)
                                         : BI->getSuccessor(0);
          if (!AuthorityBody)
            continue;
          bool DominatesAllRiskLoops = true;
          for (BasicBlock *Header : Report.LoopHeaders) {
            if (!Header || !DT.dominates(AuthorityBody, Header)) {
              DominatesAllRiskLoops = false;
              break;
            }
          }
          if (DominatesAllRiskLoops) {
            Report.GuardedByAuthorityLane = true;
            break;
          }
        }
        if (Report.GuardedByAuthorityLane)
          break;
      }
    }

    if (!Report.IsBarrierFree)
      Report.Reasons.push_back("has_reachable_barrier");
    if (!Report.HasSharedReads)
      Report.Reasons.push_back("no_shared_reads");
    if (!Report.HasSharedWrites)
      Report.Reasons.push_back("no_shared_writes");
    if (!Report.HasCrossLaneSharedAccess)
      Report.Reasons.push_back("no_cross_lane_shared_access");
    if (!Report.HasLoopContainedSharedUpdate)
      Report.Reasons.push_back("no_loop_contained_shared_update");

    auto SetResidualReason = [&](StringRef R) {
      if (Report.ResidualRiskReason.empty())
        Report.ResidualRiskReason = R.str();
    };
    bool Mitigated = false;
    if (RawRisk) {
      Mitigated = Report.GuardedByAuthorityLane &&
                  Report.HasEarlyAuthorityReturnShape &&
                  (!Report.HasCarryPreloadSubregion ||
                   Report.HasMicroKernelRewrite) &&
                  (!Report.NeedsSpillFanout || Report.HasSpillFanout);
      if (!Report.GuardedByAuthorityLane)
        SetResidualReason("authority_guard_missing");
      else if (!Report.HasEarlyAuthorityReturnShape)
        SetResidualReason("early_authority_shape_missing");
      else if (Report.HasCarryPreloadSubregion && !Report.HasMicroKernelRewrite)
        SetResidualReason("carry_preload_rewrite_missing");
      else if (Report.NeedsSpillFanout && !Report.HasSpillFanout)
        SetResidualReason("spill_fanout_missing");
      else if (!Mitigated)
        SetResidualReason("risk_persists_after_lowering");
    }

    Report.HasRisk = RawRisk && !Mitigated;
    if (Report.GuardedByAuthorityLane)
      Report.Reasons.push_back("authority_guard_present");
    if (Report.HasEarlyAuthorityReturnShape)
      Report.Reasons.push_back("authority_early_return_shape_present");
    if (Report.HasCarryPreloadSubregion)
      Report.Reasons.push_back("carry_preload_subregion_detected");
    if (Report.HasMicroKernelRewrite)
      Report.Reasons.push_back("carry_preload_subregion_rewritten");
    if (HasBacksolveGlobalRecurrence)
      Report.Reasons.push_back("global_backsolve_recurrence_detected");
    if (Report.NeedsSpillFanout)
      Report.Reasons.push_back("spill_fanout_required");
    if (Report.HasSpillFanout)
      Report.Reasons.push_back("spill_fanout_applied");
    if (Report.HasRisk)
      Report.Reasons.push_back("cooperative_phase_risk_detected");
    else if (RawRisk)
      Report.Reasons.push_back("cooperative_phase_risk_mitigated");
    else
      Report.Reasons.push_back("cooperative_phase_risk_not_detected");

    std::string ReasonSummary;
    for (unsigned I = 0; I < Report.Reasons.size(); ++I) {
      if (I)
        ReasonSummary += "|";
      ReasonSummary += Report.Reasons[I];
    }
    if (ReasonSummary.empty())
      ReasonSummary = "none";

    errs() << "deepFission: [DF][PHASE] event=detect.summary func="
           << F->getName() << " stage=" << Stage
           << " barrier_free=" << (Report.IsBarrierFree ? 1 : 0)
           << " shared_reads=" << Report.SharedReads.size()
           << " shared_writes=" << Report.SharedWrites.size()
           << " cross_lane=" << (Report.HasCrossLaneSharedAccess ? 1 : 0)
           << " loop_update=" << (Report.HasLoopContainedSharedUpdate ? 1 : 0)
           << " loop_headers=" << Report.LoopHeaders.size()
           << " axis=" << authorityAxisKindName(Report.Axis)
           << " axis_modulo=" << Report.AxisModulo
           << " guarded=" << (Report.GuardedByAuthorityLane ? 1 : 0)
           << " early_authority_shape="
           << (Report.HasEarlyAuthorityReturnShape ? 1 : 0)
           << " carry_preload=" << (Report.HasCarryPreloadSubregion ? 1 : 0)
           << " subregion_rewritten=" << (Report.HasMicroKernelRewrite ? 1 : 0)
           << " spill_fanout_needed=" << (Report.NeedsSpillFanout ? 1 : 0)
           << " spill_fanout=" << (Report.HasSpillFanout ? 1 : 0)
           << " residual_reason="
           << (Report.ResidualRiskReason.empty() ? StringRef("none")
                                                 : StringRef(Report.ResidualRiskReason))
           << " risk=" << (Report.HasRisk ? 1 : 0)
           << " reasons=" << ReasonSummary << "\n";
    for (BasicBlock *Header : Report.LoopHeaders) {
      errs() << "deepFission: [DF][PHASE] event=detect.loop func="
             << F->getName() << " stage=" << Stage
             << " header=" << (Header ? Header->getName() : StringRef("<null>"))
             << "\n";
    }
    return Report;
  }

  bool canSpillLiveIn(Value *V, Function *F, Instruction *SpillPt,
                      DominatorTree &DT) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !F || !SpillPt || I->getFunction() != F)
      return false;
    if (isa<AllocaInst>(I))
      return false;
    Type *Ty = I->getType();
    if (!Ty || !Ty->isFirstClassType() || Ty->isVoidTy() || Ty->isTokenTy() ||
        Ty->isLabelTy() || Ty->isMetadataTy())
      return false;
    if (isa<PHINode>(I))
      return false;
    return DT.dominates(I, SpillPt);
  }

  GlobalVariable *createSpillGlobal(Module &M, Function *F, Type *ElemTy,
                                    unsigned SpillIndex) {
    auto *SpillTy = ArrayType::get(ElemTy, 1024);
    std::string BaseName =
        getDeepFissionBaseName(F) + ".df.spill." + std::to_string(SpillIndex);
    std::string Name = BaseName;
    unsigned Suffix = 0;
    while (M.getNamedValue(Name))
      Name = BaseName + "." + std::to_string(++Suffix);

    return new GlobalVariable(M, SpillTy, false, GlobalValue::InternalLinkage,
                              Constant::getNullValue(SpillTy), Name, nullptr,
                              GlobalVariable::GeneralDynamicTLSModel, 0);
  }

  Value *buildSpillSlot(IRBuilder<> &B, GlobalVariable *SpillGV,
                        Value *LinearTid) {
    if (!SpillGV || !LinearTid)
      return nullptr;
    Value *Zero = ConstantInt::get(Type::getInt64Ty(B.getContext()), 0);
    Value *Index = LinearTid;
    if (!Index->getType()->isIntegerTy(64))
      Index = B.CreateZExtOrTrunc(Index, Type::getInt64Ty(B.getContext()),
                                  "df.spill.idx");
    Value *Indices[] = {Zero, Index};
    return B.CreateInBoundsGEP(SpillGV->getValueType(), SpillGV, Indices,
                               "df.spill.slot");
  }

  bool canMaterializeLiveIn(Value *V, DenseSet<Value *> &Visiting) {
    if (!V)
      return false;
    if (isa<Argument>(V) || isa<Constant>(V) || isa<GlobalValue>(V))
      return true;

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return false;

    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (!isSafeInvariantLoadForLiveIn(LI))
        return false;
      if (!Visiting.insert(V).second)
        return false;
      if (!canMaterializeLiveIn(LI->getPointerOperand(), Visiting)) {
        Visiting.erase(V);
        return false;
      }
      Visiting.erase(V);
      return true;
    }

    if (isa<PHINode>(I) || isa<AllocaInst>(I) || isa<StoreInst>(I) ||
        isa<CallInst>(I))
      return false;

    if (!Visiting.insert(V).second)
      return false;
    for (Value *Op : I->operands()) {
      if (!canMaterializeLiveIn(Op, Visiting)) {
        Visiting.erase(V);
        return false;
      }
    }
    Visiting.erase(V);
    return true;
  }

  SmallVector<BoundaryValueInfo, 8>
  collectBoundaryValueInfos(Function *F, Instruction *SpillPt,
                            ArrayRef<Value *> LiveIns, DominatorTree &DT,
                            unsigned &UnresolvedLiveIns) {
    SmallVector<BoundaryValueInfo, 8> Infos;
    for (Value *V : LiveIns) {
      if (isa<Argument>(V) || isa<Constant>(V) || isa<GlobalValue>(V))
        continue;

      DenseSet<Value *> Visiting;
      BoundaryValueInfo Info;
      Info.Value = V;
      Info.StorageType = getLiveInSpillStorageType(V);
      Info.CanonicalUse = findCanonicalBoundaryUse(V, F);
      Info.CarrierAlloca = dyn_cast<AllocaInst>(V);

      if (canMaterializeLiveIn(V, Visiting)) {
        Info.Kind = BoundaryValueKind::Rematerialize;
      } else if (canSpillAllocaSeedLiveIn(V, F, SpillPt, DT)) {
        Info.Kind = isReg2MemCarrierAlloca(Info.CarrierAlloca)
                        ? BoundaryValueKind::SpillReg2MemCarrier
                        : BoundaryValueKind::SpillSimpleAllocaSeed;
      } else if (canSpillLiveIn(V, F, SpillPt, DT)) {
        Info.Kind = BoundaryValueKind::SpillScalar;
      } else {
        ++UnresolvedLiveIns;
        errs() << "deepFission: unresolved live-in " << *V << "\n";
        continue;
      }

      if (Info.Kind != BoundaryValueKind::Rematerialize && !Info.StorageType) {
        ++UnresolvedLiveIns;
        errs() << "deepFission: unresolved spill storage type for " << *V
               << "\n";
        continue;
      }

      errs() << "deepFission: boundary live-in " << *V
             << " kind=" << boundaryValueKindName(Info.Kind) << "\n";
      if (Info.Kind == BoundaryValueKind::SpillReg2MemCarrier) {
        errs() << "deepFission: classified boundary carrier " << *V
               << " as spill-reg2mem-carrier\n";
      }
      Infos.push_back(Info);
    }
    return Infos;
  }

  Value *materializeLiveInInPlace(Value *V, IRBuilder<> &B,
                                  DenseMap<Value *, Value *> &Cache,
                                  DominatorTree &DT,
                                  Instruction *DominancePt = nullptr) {
    if (!V)
      return nullptr;
    auto CacheIt = Cache.find(V);
    if (CacheIt != Cache.end())
      return CacheIt->second;

    Instruction *InsertPt = DominancePt;
    if (!InsertPt && B.GetInsertBlock()) {
      BasicBlock::iterator It = B.GetInsertPoint();
      if (It != B.GetInsertBlock()->end())
        InsertPt = &*It;
      else
        InsertPt = B.GetInsertBlock()->getTerminator();
    }
    if (auto *A = dyn_cast<Argument>(V))
      return Cache[V] = A;
    if (isa<Constant>(V) || isa<GlobalValue>(V))
      return Cache[V] = V;

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return nullptr;
    if (InsertPt && DT.dominates(I, InsertPt))
      return Cache[V] = I;

    auto MaterializeOperand = [&](Value *Op) -> Value * {
      return materializeLiveInInPlace(Op, B, Cache, DT, DominancePt);
    };

    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (!isSafeInvariantLoadForLiveIn(LI))
        return nullptr;
      Value *Ptr = MaterializeOperand(LI->getPointerOperand());
      if (!Ptr)
        return nullptr;
      auto *NewLoad = B.CreateLoad(LI->getType(), Ptr, "df.livein");
      NewLoad->setAlignment(LI->getAlignment());
      return Cache[V] = NewLoad;
    }
    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      Value *L = MaterializeOperand(BO->getOperand(0));
      Value *R = MaterializeOperand(BO->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateBinOp(BO->getOpcode(), L, R, "df.livein");
    }
    if (auto *IC = dyn_cast<ICmpInst>(I)) {
      Value *L = MaterializeOperand(IC->getOperand(0));
      Value *R = MaterializeOperand(IC->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateICmp(IC->getPredicate(), L, R, "df.livein");
    }
    if (auto *FC = dyn_cast<FCmpInst>(I)) {
      Value *L = MaterializeOperand(FC->getOperand(0));
      Value *R = MaterializeOperand(FC->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateFCmp(FC->getPredicate(), L, R, "df.livein");
    }
    if (auto *SI = dyn_cast<SelectInst>(I)) {
      Value *C = MaterializeOperand(SI->getCondition());
      Value *T = MaterializeOperand(SI->getTrueValue());
      Value *Fv = MaterializeOperand(SI->getFalseValue());
      if (!C || !T || !Fv)
        return nullptr;
      return Cache[V] = B.CreateSelect(C, T, Fv, "df.livein");
    }
    if (auto *Cast = dyn_cast<CastInst>(I)) {
      Value *Op = MaterializeOperand(Cast->getOperand(0));
      if (!Op)
        return nullptr;
      return Cache[V] =
                 B.CreateCast(Cast->getOpcode(), Op, Cast->getType(),
                              "df.livein");
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      Value *Base = MaterializeOperand(GEP->getPointerOperand());
      if (!Base)
        return nullptr;
      SmallVector<Value *, 8> Indices;
      for (Value *IdxV : GEP->indices()) {
        Value *Idx = MaterializeOperand(IdxV);
        if (!Idx)
          return nullptr;
        Indices.push_back(Idx);
      }
      if (GEP->isInBounds()) {
        return Cache[V] =
                   B.CreateInBoundsGEP(GEP->getSourceElementType(), Base,
                                       Indices, "df.livein");
      }
      return Cache[V] =
                 B.CreateGEP(GEP->getSourceElementType(), Base, Indices,
                             "df.livein");
    }
    return nullptr;
  }

  Value *resolveBoundaryStoredValue(Value *Stored, BasicBlock *Pred,
                                    IRBuilder<> &B, DominatorTree &DT) {
    if (!Stored || !Pred || isa<UndefValue>(Stored))
      return nullptr;

    Instruction *PredTerm = Pred->getTerminator();
    if (!PredTerm)
      return nullptr;

    if (isa<Argument>(Stored) || isa<Constant>(Stored) || isa<GlobalValue>(Stored))
      return Stored;

    auto *StoredI = dyn_cast<Instruction>(Stored);
    if (!StoredI)
      return nullptr;
    if (StoredI->getFunction() == Pred->getParent() &&
        DT.dominates(StoredI, PredTerm))
      return StoredI;

    DenseSet<Value *> Visiting;
    DenseMap<Value *, Value *> Cache;
    if (canMaterializeLiveIn(Stored, Visiting))
      return materializeLiveInInPlace(Stored, B, Cache, DT, PredTerm);
    return nullptr;
  }

  Value *resolveCarrierScalarOnEdge(AllocaInst *AI, BasicBlock *Pred,
                                    BasicBlock *Succ, IRBuilder<> &B,
                                    DominatorTree &DT) {
    if (!AI || !Pred || !Succ)
      return nullptr;

    auto TryResolveInBlock = [&](BasicBlock *Current,
                                 Instruction *Before) -> Value * {
      Instruction *Access = findLastCarrierAccessBefore(AI, Current, Before);
      if (!Access)
        return nullptr;

      if (auto *SI = dyn_cast<StoreInst>(Access)) {
        Value *Resolved =
            resolveBoundaryStoredValue(SI->getValueOperand(), Pred, B, DT);
        if (!Resolved || isa<UndefValue>(Resolved))
          return nullptr;
        errs() << "deepFission: resolved carrier scalar on edge "
               << Pred->getName() << " -> " << Succ->getName() << " for "
               << AI->getName() << " as " << *Resolved << "\n";
        return Resolved;
      }

      auto *LI = dyn_cast<LoadInst>(Access);
      if (!LI)
        return nullptr;
      if (LI->getFunction() == Pred->getParent() &&
          DT.dominates(LI, Pred->getTerminator())) {
        errs() << "deepFission: resolved carrier scalar on edge "
               << Pred->getName() << " -> " << Succ->getName() << " for "
               << AI->getName() << " from dominating reload " << *LI << "\n";
        return LI;
      }
      return nullptr;
    };

    SmallPtrSet<BasicBlock *, 8> Visited;
    BasicBlock *Current = Pred;
    while (Current && Visited.insert(Current).second) {
      Instruction *Before =
          (Current == Pred) ? Current->getTerminator() : nullptr;
      if (Value *Resolved = TryResolveInBlock(Current, Before))
        return Resolved;

      BasicBlock *SinglePred = Current->getSinglePredecessor();
      if (!SinglePred || SinglePred->getTerminator()->getNumSuccessors() != 1)
        break;
      Current = SinglePred;
    }

    DomTreeNode *Node = DT.getNode(Pred);
    while (Node) {
      BasicBlock *DomBB = Node->getBlock();
      if (!DomBB || !Visited.insert(DomBB).second) {
        Node = Node->getIDom();
        continue;
      }
      if (Value *Resolved = TryResolveInBlock(DomBB, nullptr))
        return Resolved;
      Node = Node->getIDom();
    }
    return nullptr;
  }

  Value *computeBoundaryOutgoingValue(const BoundaryValueInfo &Info,
                                      BasicBlock *Pred, BasicBlock *Succ,
                                      IRBuilder<> &B, DominatorTree &DT) {
    if (!Info.Value || !Pred || !Pred->getTerminator())
      return nullptr;

    DenseMap<Value *, Value *> Cache;
    Instruction *PredTerm = Pred->getTerminator();
    switch (Info.Kind) {
    case BoundaryValueKind::Rematerialize:
      return materializeLiveInInPlace(Info.Value, B, Cache, DT, PredTerm);
    case BoundaryValueKind::SpillSimpleAllocaSeed:
    case BoundaryValueKind::SpillReg2MemCarrier:
      return resolveCarrierScalarOnEdge(
          Info.CarrierAlloca ? Info.CarrierAlloca : dyn_cast<AllocaInst>(Info.Value),
          Pred, Succ, B, DT);
    case BoundaryValueKind::SpillScalar:
      break;
    }

    auto *I = dyn_cast<Instruction>(Info.Value);
    if (!I)
      return nullptr;

    if (auto *PN = dyn_cast<PHINode>(I)) {
      int IncomingIdx = PN->getBasicBlockIndex(Pred);
      if (IncomingIdx < 0)
        return nullptr;
      Value *Incoming = PN->getIncomingValue(IncomingIdx);
      if (isa<UndefValue>(Incoming))
        return nullptr;
      if (auto *IncomingI = dyn_cast<Instruction>(Incoming)) {
        if (DT.dominates(IncomingI, Pred->getTerminator()))
          return IncomingI;
      } else if (isa<Argument>(Incoming) || isa<Constant>(Incoming) ||
                 isa<GlobalValue>(Incoming)) {
        return Incoming;
      }
      DenseSet<Value *> Visiting;
      if (canMaterializeLiveIn(Incoming, Visiting))
        return materializeLiveInInPlace(Incoming, B, Cache, DT, PredTerm);
      return nullptr;
    }

    if (auto *LI = dyn_cast<LoadInst>(I)) {
      Value *Ptr = LI->getPointerOperand()->stripPointerCasts();
      if (auto *AI = dyn_cast<AllocaInst>(Ptr)) {
        if (AI->getFunction() != Pred->getParent() ||
            AI->getParent() != &Pred->getParent()->getEntryBlock())
          return nullptr;
        auto *Reload = B.CreateLoad(LI->getType(), AI, "df.boundary.reload");
        Reload->setAlignment(LI->getAlignment());
        return Reload;
      }
    }

    if (DT.dominates(I, Pred->getTerminator()))
      return I;

    DenseSet<Value *> Visiting;
    if (canMaterializeLiveIn(Info.Value, Visiting))
      return materializeLiveInInPlace(Info.Value, B, Cache, DT, PredTerm);

    return nullptr;
  }

  BoundaryExitPlan buildBoundaryExitBlock(
      Function *F, const SmallVectorImpl<BoundaryValueInfo> &BoundaryValues,
      SmallVectorImpl<BoundaryEdgeValue> &BoundaryEdges, DominatorTree &DT) {
    BoundaryExitPlan Plan;
    if (!F || BoundaryEdges.empty())
      return Plan;

    BasicBlock *InsertBefore = nullptr;
    for (BoundaryEdgeValue &Edge : BoundaryEdges) {
      if (Edge.OriginalSucc) {
        InsertBefore = Edge.OriginalSucc;
        break;
      }
    }
    Plan.ExitBlock = BasicBlock::Create(F->getContext(), "df.boundary.exit", F,
                                        InsertBefore);

    for (const BoundaryValueInfo &Info : BoundaryValues) {
      if (Info.Kind == BoundaryValueKind::Rematerialize)
        continue;
      Plan.ExportPhis.push_back(PHINode::Create(Info.StorageType,
                                                BoundaryEdges.size(),
                                                "df.boundary.value",
                                                Plan.ExitBlock));
    }

    for (BoundaryEdgeValue &Edge : BoundaryEdges) {
      Edge.EdgeBlock = BasicBlock::Create(F->getContext(), "df.boundary.edge", F,
                                          Plan.ExitBlock);
    }

    for (BoundaryEdgeValue &Edge : BoundaryEdges) {
      IRBuilder<> EdgeBuilder(Edge.EdgeBlock);
      unsigned SpillIdx = 0;
      for (const BoundaryValueInfo &Info : BoundaryValues) {
        if (Info.Kind == BoundaryValueKind::Rematerialize)
          continue;
        Value *Outgoing =
            computeBoundaryOutgoingValue(Info, Edge.Pred, Edge.OriginalSucc,
                                         EdgeBuilder, DT);
        if (!Outgoing || isa<UndefValue>(Outgoing)) {
          errs() << "deepFission: failed to materialize boundary value "
                 << *Info.Value << " on edge " << Edge.Pred->getName() << " -> "
                 << Edge.OriginalSucc->getName() << "\n";
          for (BoundaryEdgeValue &CleanupEdge : BoundaryEdges) {
            if (CleanupEdge.EdgeBlock)
              CleanupEdge.EdgeBlock->eraseFromParent();
          }
          if (Plan.ExitBlock)
            Plan.ExitBlock->eraseFromParent();
          Plan.ExitBlock = nullptr;
          return Plan;
        }
        Plan.ExportPhis[SpillIdx++]->addIncoming(Outgoing, Edge.EdgeBlock);
        Edge.ExportedValues.push_back(Outgoing);
      }
      EdgeBuilder.CreateBr(Plan.ExitBlock);
    }

    for (BoundaryEdgeValue &Edge : BoundaryEdges)
      Plan.EdgeValues.push_back(Edge);

    for (BoundaryEdgeValue &Edge : BoundaryEdges)
      Edge.Pred->getTerminator()->setSuccessor(Edge.SuccessorIndex,
                                               Edge.EdgeBlock);

    IRBuilder<> ExitBuilder(Plan.ExitBlock);
    Value *LinearTid = buildLinearThreadIndex(ExitBuilder, F);
    if (!LinearTid) {
      Plan.ExitBlock = nullptr;
      return Plan;
    }

    unsigned SpillIdx = 0;
    for (const BoundaryValueInfo &Info : BoundaryValues) {
      if (Info.Kind == BoundaryValueKind::Rematerialize)
        continue;
      PHINode *Phi = Plan.ExportPhis[SpillIdx++];
      for (unsigned IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues();
           ++IncomingIdx) {
        if (isa<UndefValue>(Phi->getIncomingValue(IncomingIdx))) {
          errs() << "deepFission: refusing boundary export with undef incoming "
                 << "for " << *Info.Value << "\n";
          Plan.ExitBlock = nullptr;
          return Plan;
        }
      }
      Value *Slot = buildSpillSlot(ExitBuilder, Info.SpillGlobal, LinearTid);
      if (!Slot) {
        Plan.ExitBlock = nullptr;
        return Plan;
      }
      ExitBuilder.CreateStore(Phi, Slot);
    }

    ReturnInst *RI = nullptr;
    if (F->getReturnType()->isVoidTy())
      RI = ExitBuilder.CreateRetVoid();
    else
      RI = ExitBuilder.CreateRet(UndefValue::get(F->getReturnType()));
    markSyntheticReturn(RI, true);
    return Plan;
  }

  Value *materializeLiveInAtCallSite(Value *V, CallInst *CallI, IRBuilder<> &B,
                                     DenseMap<Value *, Value *> &Cache) {
    if (!V)
      return nullptr;
    if (Cache.count(V))
      return Cache[V];

    if (Argument *A = dyn_cast<Argument>(V))
      return Cache[V] = CallI->getArgOperand(A->getArgNo());
    if (isa<Constant>(V) || isa<GlobalValue>(V))
      return Cache[V] = V;

    Instruction *I = dyn_cast<Instruction>(V);
    if (!I)
      return nullptr;

    auto MaterializeOperand = [&](Value *Op) -> Value * {
      return materializeLiveInAtCallSite(Op, CallI, B, Cache);
    };

    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (!isSafeInvariantLoadForLiveIn(LI))
        return nullptr;
      Value *Ptr = MaterializeOperand(LI->getPointerOperand());
      if (!Ptr)
        return nullptr;
      auto *NewLoad = B.CreateLoad(LI->getType(), Ptr, "df.livein");
      NewLoad->setAlignment(LI->getAlignment());
      return Cache[V] = NewLoad;
    }
    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      Value *L = MaterializeOperand(BO->getOperand(0));
      Value *R = MaterializeOperand(BO->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateBinOp(BO->getOpcode(), L, R, "df.livein");
    }
    if (auto *IC = dyn_cast<ICmpInst>(I)) {
      Value *L = MaterializeOperand(IC->getOperand(0));
      Value *R = MaterializeOperand(IC->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateICmp(IC->getPredicate(), L, R, "df.livein");
    }
    if (auto *FC = dyn_cast<FCmpInst>(I)) {
      Value *L = MaterializeOperand(FC->getOperand(0));
      Value *R = MaterializeOperand(FC->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateFCmp(FC->getPredicate(), L, R, "df.livein");
    }
    if (auto *SI = dyn_cast<SelectInst>(I)) {
      Value *C = MaterializeOperand(SI->getCondition());
      Value *T = MaterializeOperand(SI->getTrueValue());
      Value *F = MaterializeOperand(SI->getFalseValue());
      if (!C || !T || !F)
        return nullptr;
      return Cache[V] = B.CreateSelect(C, T, F, "df.livein");
    }
    if (auto *Cast = dyn_cast<CastInst>(I)) {
      Value *Op = MaterializeOperand(Cast->getOperand(0));
      if (!Op)
        return nullptr;
      return Cache[V] =
                 B.CreateCast(Cast->getOpcode(), Op, Cast->getType(),
                              "df.livein");
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      Value *Base = MaterializeOperand(GEP->getPointerOperand());
      if (!Base)
        return nullptr;
      SmallVector<Value *, 8> Indices;
      for (Value *IdxV : GEP->indices()) {
        Value *Idx = MaterializeOperand(IdxV);
        if (!Idx)
          return nullptr;
        Indices.push_back(Idx);
      }
      if (GEP->isInBounds()) {
        return Cache[V] =
                   B.CreateInBoundsGEP(GEP->getSourceElementType(), Base,
                                       Indices, "df.livein");
      }
      return Cache[V] =
                 B.CreateGEP(GEP->getSourceElementType(), Base, Indices,
                             "df.livein");
    }
    return nullptr;
  }

  Value *materializeLiveInInFunction(Value *V, Function *F,
                                     ValueToValueMapTy &VMap, IRBuilder<> &B,
                                     DenseMap<Value *, Value *> &Cache) {
    if (!V)
      return nullptr;
    auto CacheIt = Cache.find(V);
    if (CacheIt != Cache.end())
      return CacheIt->second;

    if (auto *A = dyn_cast<Argument>(V)) {
      auto MapIt = VMap.find(A);
      if (MapIt == VMap.end())
        return nullptr;
      return Cache[V] = MapIt->second;
    }
    if (isa<Constant>(V) || isa<GlobalValue>(V))
      return Cache[V] = V;

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return nullptr;

    auto MaterializeOperand = [&](Value *Op) -> Value * {
      return materializeLiveInInFunction(Op, F, VMap, B, Cache);
    };

    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (!isSafeInvariantLoadForLiveIn(LI))
        return nullptr;
      Value *Ptr = MaterializeOperand(LI->getPointerOperand());
      if (!Ptr)
        return nullptr;
      auto *NewLoad = B.CreateLoad(LI->getType(), Ptr, "df.livein");
      NewLoad->setAlignment(LI->getAlignment());
      return Cache[V] = NewLoad;
    }
    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      Value *L = MaterializeOperand(BO->getOperand(0));
      Value *R = MaterializeOperand(BO->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateBinOp(BO->getOpcode(), L, R, "df.livein");
    }
    if (auto *IC = dyn_cast<ICmpInst>(I)) {
      Value *L = MaterializeOperand(IC->getOperand(0));
      Value *R = MaterializeOperand(IC->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateICmp(IC->getPredicate(), L, R, "df.livein");
    }
    if (auto *FC = dyn_cast<FCmpInst>(I)) {
      Value *L = MaterializeOperand(FC->getOperand(0));
      Value *R = MaterializeOperand(FC->getOperand(1));
      if (!L || !R)
        return nullptr;
      return Cache[V] = B.CreateFCmp(FC->getPredicate(), L, R, "df.livein");
    }
    if (auto *SI = dyn_cast<SelectInst>(I)) {
      Value *C = MaterializeOperand(SI->getCondition());
      Value *T = MaterializeOperand(SI->getTrueValue());
      Value *Fv = MaterializeOperand(SI->getFalseValue());
      if (!C || !T || !Fv)
        return nullptr;
      return Cache[V] = B.CreateSelect(C, T, Fv, "df.livein");
    }
    if (auto *Cast = dyn_cast<CastInst>(I)) {
      Value *Op = MaterializeOperand(Cast->getOperand(0));
      if (!Op)
        return nullptr;
      return Cache[V] =
                 B.CreateCast(Cast->getOpcode(), Op, Cast->getType(),
                              "df.livein");
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      Value *Base = MaterializeOperand(GEP->getPointerOperand());
      if (!Base)
        return nullptr;
      SmallVector<Value *, 8> Indices;
      for (Value *IdxV : GEP->indices()) {
        Value *Idx = MaterializeOperand(IdxV);
        if (!Idx)
          return nullptr;
        Indices.push_back(Idx);
      }
      if (GEP->isInBounds())
        return Cache[V] = B.CreateInBoundsGEP(GEP->getSourceElementType(), Base,
                                              Indices, "df.livein");
      return Cache[V] =
                 B.CreateGEP(GEP->getSourceElementType(), Base, Indices,
                             "df.livein");
    }

    return nullptr;
  }

  bool promoteKernelAllocasToSSA(Function *F) {
    if (!F || F->isDeclaration())
      return false;
    SmallVector<AllocaInst *, 32> Promotable;
    for (Instruction &I : F->getEntryBlock()) {
      auto *AI = dyn_cast<AllocaInst>(&I);
      if (AI && isAllocaPromotable(AI))
        Promotable.push_back(AI);
    }
    if (Promotable.empty())
      return false;

    DominatorTree DT(*F);
    PromoteMemToReg(Promotable, DT);
    errs() << "deepFission: mem2reg promoted allocas in " << F->getName()
           << ", count=" << Promotable.size() << "\n";
    return true;
  }

  bool demoteKernelPhisToStack(Function *F) {
    if (!F || F->isDeclaration())
      return false;
    SmallVector<PHINode *, 32> Phis;
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break;
        Phis.push_back(PN);
      }
    }
    if (Phis.empty())
      return false;

    Instruction *AllocaPt =
        F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
    if (!AllocaPt)
      AllocaPt = F->getEntryBlock().getTerminator();

    for (PHINode *PN : Phis)
      DemotePHIToStack(PN, AllocaPt);

    errs() << "deepFission: demoted phi nodes in " << F->getName()
           << ", count=" << Phis.size() << "\n";
    return true;
  }

  bool canonicalizeKernelEarlyReturns(Function *F) {
    if (!F || F->isDeclaration())
      return false;

    PostDominatorTree PDT(*F);
    BasicBlock *EntryBB = &F->getEntryBlock();
    SmallVector<ReturnInst *, 8> EarlyReturns;
    for (Instruction &I : instructions(F)) {
      auto *RI = dyn_cast<ReturnInst>(&I);
      if (!RI || isSyntheticReturn(RI))
        continue;
      if (isSplitStageFunction(F) &&
          (blockPerformsObservableStageWork(RI->getParent()) ||
           returnHasObservableWorkOnIncomingRegion(RI))) {
        markSyntheticReturn(RI, true);
        continue;
      }
      if (!PDT.dominates(RI->getParent(), EntryBB))
        EarlyReturns.push_back(RI);
    }

    if (EarlyReturns.empty())
      return false;

    BasicBlock *ExitBB =
        BasicBlock::Create(F->getContext(), "df.kernel.early.exit", F);
    IRBuilder<> ExitBuilder(ExitBB);
    PHINode *RetPhi = nullptr;
    if (!F->getReturnType()->isVoidTy()) {
      RetPhi = ExitBuilder.CreatePHI(F->getReturnType(), EarlyReturns.size(),
                                     "df.kernel.ret");
      markLaneTerminatingReturn(ExitBuilder.CreateRet(RetPhi));
    } else {
      markLaneTerminatingReturn(ExitBuilder.CreateRetVoid());
    }
    markLaneTerminatingExitBlock(ExitBB);

    bool Changed = false;
    for (ReturnInst *RI : EarlyReturns) {
      if (!RI || RI->getParent() == ExitBB)
        continue;
      BasicBlock *PredBB = RI->getParent();
      SmallPtrSet<BasicBlock *, 8> ExitRegion;
      collectLaneTerminatingExitPredecessors(PredBB, ExitRegion);
      SmallVector<BasicBlock *, 8> RegionBlocks(ExitRegion.begin(),
                                                ExitRegion.end());
      for (BasicBlock *RegionBB : RegionBlocks) {
        SmallVector<BasicBlock *, 8> Preds(predecessors(RegionBB));
        for (BasicBlock *BoundaryPred : Preds) {
          if (ExitRegion.count(BoundaryPred))
            continue;
        }
      }
      IRBuilder<> B(RI);
      if (RetPhi)
        RetPhi->addIncoming(RI->getReturnValue(), PredBB);
      B.CreateBr(ExitBB);
      RI->eraseFromParent();
      Changed = true;
    }

    if (Changed) {
      (void)simplifyDeadControlFlow(F);
      (void)pruneUnreachableBlocks(F);
      errs() << "deepFission: canonicalized early returns in " << F->getName()
             << ", count=" << EarlyReturns.size() << "\n";
    }
    return Changed;
  }

  void normalizeKernelForSplit(Function *F) {
    if (!F || F->isDeclaration())
      return;
    canonicalizeKernelEarlyReturns(F);
    promoteKernelAllocasToSSA(F);
    demoteKernelPhisToStack(F);
    simplifyDeadControlFlow(F);
  }

  bool stabilizeBarrierFreeWorkStage(Function *F) {
    if (!F || F->isDeclaration())
      return false;
    errs() << "deepFission: [DF][RESUME] event=workstage.stabilize.start func="
           << F->getName() << " stage=" << getStageIndexForLog(F) << "\n";
    (void)simplifyDeadControlFlow(F);
    (void)pruneUnreachableBlocks(F);
    (void)repairNonDominatingAllocaLoads(F);
    if (!repairReachableStageLocalCarrierSeeds(F))
      return false;
    if (!stabilizeBarrierFreeStageForCBE(F, false))
      return false;
    if (!validateReachableCarrierSeedStores(F))
      return false;
    if (!validateFinalStageSemantics(F))
      return false;
    if (!verifyFunctionOrReport("stabilizeBarrierFreeWorkStage", F))
      return false;
    errs() << "deepFission: [DF][RESUME] event=workstage.stabilize.done func="
           << F->getName() << " stage=" << getStageIndexForLog(F) << "\n";
    return true;
  }

  void replaceTerminatorWithReturnAndRepairPhis(BasicBlock *BB,
                                                IRBuilder<> &Builder) {
    if (!BB)
      return;
    Instruction *OldTerm = BB->getTerminator();
    if (!OldTerm)
      return;

    SmallVector<BasicBlock *, 4> OldSuccs;
    for (unsigned I = 0; I < OldTerm->getNumSuccessors(); ++I)
      OldSuccs.push_back(OldTerm->getSuccessor(I));

    SmallPtrSet<BasicBlock *, 4> DedupSuccs(OldSuccs.begin(), OldSuccs.end());
    for (BasicBlock *Succ : DedupSuccs) {
      if (Succ)
        Succ->removePredecessor(BB, false);
    }

    Builder.SetInsertPoint(OldTerm);
    Type *RetTy = BB->getParent()->getReturnType();
    ReturnInst *RI = nullptr;
    if (RetTy->isVoidTy())
      RI = Builder.CreateRetVoid();
    else
      RI = Builder.CreateRet(UndefValue::get(RetTy));
    markSyntheticReturn(RI);
    OldTerm->eraseFromParent();
  }

  void sanitizeClonedEntryBlock(BasicBlock *Entry) {
    if (!Entry)
      return;

    SmallVector<Instruction *, 16> ToErase;
    for (Instruction &I : *Entry) {
      if (I.isTerminator())
        continue;
      if (isa<DbgInfoIntrinsic>(&I))
        continue;
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        Intrinsic::ID ID = II->getIntrinsicID();
        if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
          continue;
      }
      if (!I.use_empty())
        continue;
      if (I.mayHaveSideEffects())
        continue;
      ToErase.push_back(&I);
    }

    for (Instruction *I : ToErase)
      I->eraseFromParent();
  }

  Instruction *getStageEntryInsertPoint(Function *F) {
    if (!F || F->empty())
      return nullptr;
    Instruction *InsertPt =
        F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
    while (InsertPt && isa<AllocaInst>(InsertPt))
      InsertPt = InsertPt->getNextNode();
    if (!InsertPt)
      InsertPt = F->getEntryBlock().getTerminator();
    return InsertPt;
  }

  bool repairConsumerCarrierSeeds(Function *NewFunc,
                                  ArrayRef<BoundaryValueInfo> BoundaryValues,
                                  ValueToValueMapTy &VMap) {
    if (!NewFunc)
      return false;

    Instruction *InsertPt = getStageEntryInsertPoint(NewFunc);
    if (!InsertPt)
      return false;
    IRBuilder<> EntryBuilder(InsertPt);
    Value *EntryLinearTid = buildLinearThreadIndex(EntryBuilder, NewFunc);
    if (!EntryLinearTid)
      return false;

    std::set<BasicBlock *> Reachable = collectReachableBlocksAvoiding(NewFunc, nullptr);
    DenseMap<AllocaInst *, Value *> SeedValues;
    bool Ok = true;

    for (const BoundaryValueInfo &Info : BoundaryValues) {
      if (Info.Kind != BoundaryValueKind::SpillSimpleAllocaSeed &&
          Info.Kind != BoundaryValueKind::SpillReg2MemCarrier)
        continue;

      auto MapIt = VMap.find(Info.Value);
      auto *MappedAI =
          (MapIt == VMap.end()) ? nullptr : dyn_cast<AllocaInst>(MapIt->second);
      if (!MappedAI || !Info.SpillGlobal || !Info.StorageType) {
        errs() << "deepFission: failed to find mapped carrier seed for "
               << *Info.Value << " in " << NewFunc->getName() << "\n";
        Ok = false;
        continue;
      }

      Value *SeedValue = SeedValues.lookup(MappedAI);
      if (!SeedValue) {
        Value *Slot =
            buildSpillSlot(EntryBuilder, Info.SpillGlobal, EntryLinearTid);
        if (!Slot) {
          Ok = false;
          continue;
        }
        SeedValue =
            EntryBuilder.CreateLoad(Info.StorageType, Slot, "df.spill.reload");
        SeedValues[MappedAI] = SeedValue;
      }

      bool HasReachableConcreteSeed = false;
      for (User *U : MappedAI->users()) {
        auto *SI = dyn_cast<StoreInst>(U);
        if (!SI || SI->getPointerOperand()->stripPointerCasts() != MappedAI ||
            !Reachable.count(SI->getParent()))
          continue;
        Value *Stored = SI->getValueOperand();
        bool ReplaceSeed = isa<UndefValue>(Stored);
        if (!ReplaceSeed) {
          if (auto *StoredI = dyn_cast<Instruction>(Stored))
            ReplaceSeed = StoredI->getFunction() != NewFunc;
        }
        if (ReplaceSeed) {
          errs() << "deepFission: repaired consumer carrier seed in "
                 << NewFunc->getName() << " for " << MappedAI->getName()
                 << " replacing " << *Stored << "\n";
          SI->setOperand(0, SeedValue);
          Stored = SeedValue;
        }
        if (!isa<UndefValue>(Stored))
          HasReachableConcreteSeed = true;
      }

      if (!HasReachableConcreteSeed) {
        StoreInst *SeedStore = EntryBuilder.CreateStore(SeedValue, MappedAI);
        SeedStore->setAlignment(MappedAI->getAlignment());
        errs() << "deepFission: inserted canonical consumer carrier seed in "
               << NewFunc->getName() << " for " << MappedAI->getName() << "\n";
      }
    }
    return Ok;
  }

  bool repairProducerBoundaryCarrierSeeds(
      Function *F, ArrayRef<BoundaryValueInfo> BoundaryValues) {
    if (!F || F->isDeclaration())
      return false;

    Instruction *InsertPt = getStageEntryInsertPoint(F);
    if (!InsertPt)
      return false;
    IRBuilder<> EntryBuilder(InsertPt);
    Value *EntryLinearTid = buildLinearThreadIndex(EntryBuilder, F);
    if (!EntryLinearTid)
      return false;

    std::set<BasicBlock *> Reachable = collectReachableBlocksAvoiding(F, nullptr);
    DenseMap<AllocaInst *, Value *> SeedValues;
    bool Ok = true;

    for (const BoundaryValueInfo &Info : BoundaryValues) {
      if (Info.Kind != BoundaryValueKind::SpillSimpleAllocaSeed &&
          Info.Kind != BoundaryValueKind::SpillReg2MemCarrier)
        continue;

      auto *AI = Info.CarrierAlloca ? Info.CarrierAlloca
                                    : dyn_cast<AllocaInst>(Info.Value);
      if (!AI || AI->getFunction() != F || !Info.SpillGlobal ||
          !Info.StorageType) {
        errs() << "deepFission: failed to find producer carrier seed for "
               << *Info.Value << " in " << F->getName() << "\n";
        Ok = false;
        continue;
      }

      Value *SeedValue = SeedValues.lookup(AI);
      if (!SeedValue) {
        Value *Slot =
            buildSpillSlot(EntryBuilder, Info.SpillGlobal, EntryLinearTid);
        if (!Slot) {
          Ok = false;
          continue;
        }
        SeedValue =
            EntryBuilder.CreateLoad(Info.StorageType, Slot, "df.spill.reload");
        SeedValues[AI] = SeedValue;
      }

      bool HasReachableConcreteSeed = false;
      for (User *U : AI->users()) {
        auto *SI = dyn_cast<StoreInst>(U);
        if (!SI || SI->getPointerOperand()->stripPointerCasts() != AI ||
            !Reachable.count(SI->getParent()))
          continue;
        Value *Stored = SI->getValueOperand();
        if (isa<UndefValue>(Stored)) {
          errs() << "deepFission: repaired producer carrier seed in "
                 << F->getName() << " for " << AI->getName()
                 << " replacing " << *Stored << "\n";
          SI->setOperand(0, SeedValue);
          Stored = SeedValue;
        }
        if (!isa<UndefValue>(Stored))
          HasReachableConcreteSeed = true;
      }

      if (!HasReachableConcreteSeed) {
        StoreInst *SeedStore = EntryBuilder.CreateStore(SeedValue, AI);
        SeedStore->setAlignment(AI->getAlignment());
        errs() << "deepFission: inserted canonical producer carrier seed in "
               << F->getName() << " for " << AI->getName() << "\n";
      }
    }
    return Ok;
  }

  bool canCarrierStoreReachLoadBeforeOverwrite(
      AllocaInst *AI, StoreInst *SeedStore,
      const std::set<BasicBlock *> &Reachable) {
    if (!AI || !SeedStore)
      return false;

    SmallVector<std::pair<BasicBlock *, Instruction *>, 16> WorkList;
    SmallPtrSet<BasicBlock *, 32> Visited;
    WorkList.push_back({SeedStore->getParent(), SeedStore});

    while (!WorkList.empty()) {
      BasicBlock *BB = WorkList.back().first;
      Instruction *StartAfter = WorkList.back().second;
      WorkList.pop_back();

      if (!BB || !Reachable.count(BB))
        continue;

      BasicBlock::iterator It =
          StartAfter ? std::next(StartAfter->getIterator()) : BB->begin();
      for (; It != BB->end(); ++It) {
        Instruction &I = *It;
        if (isa<DbgInfoIntrinsic>(&I))
          continue;
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->getPointerOperand()->stripPointerCasts() == AI)
            return true;
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->getPointerOperand()->stripPointerCasts() == AI)
            goto NextPath;
        }
      }

      if (!Visited.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB)) {
        if (Reachable.count(Succ))
          WorkList.push_back({Succ, nullptr});
      }

    NextPath:;
    }

    return false;
  }

  SmallVector<CarrierSeedStoreIssue, 8>
  collectReachableCarrierSeedStoreIssues(Function *F) {
    SmallVector<CarrierSeedStoreIssue, 8> Issues;
    if (!F || F->isDeclaration() || F->empty())
      return Issues;

    std::set<BasicBlock *> Reachable = collectReachableBlocksAvoiding(F, nullptr);
    for (BasicBlock *BB : Reachable) {
      for (Instruction &I : *BB) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI || !isa<UndefValue>(SI->getValueOperand()))
          continue;
        auto *AI =
            dyn_cast<AllocaInst>(SI->getPointerOperand()->stripPointerCasts());
        if (!AI || AI->getParent() != &F->getEntryBlock() ||
            !isReg2MemCarrierAlloca(AI))
          continue;

        CarrierSeedStoreIssue Issue;
        Issue.CarrierAlloca = AI;
        Issue.BadStore = SI;
        Issue.NearestAccess = findLastCarrierAccessBefore(AI, BB, SI);
        Issue.ReachesLoadBeforeOverwrite =
            canCarrierStoreReachLoadBeforeOverwrite(AI, SI, Reachable);
        Issues.push_back(Issue);
      }
    }
    return Issues;
  }

  Value *resolveStageLocalCarrierSeed(AllocaInst *AI, StoreInst *BadStore,
                                      DominatorTree *DT, IRBuilder<> &B) {
    if (!AI || !BadStore)
      return nullptr;

    Function *F = AI->getFunction();
    if (!F)
      return nullptr;
    DominatorTree LocalDT(*F);
    DominatorTree &UseDT = DT ? *DT : LocalDT;

    auto TryResolveInBlock = [&](BasicBlock *Current,
                                 Instruction *Before) -> Value * {
      Instruction *Access = findLastCarrierAccessBefore(AI, Current, Before);
      if (!Access)
        return nullptr;

      if (auto *SI = dyn_cast<StoreInst>(Access)) {
        if (isa<UndefValue>(SI->getValueOperand()))
          return nullptr;
        DenseMap<Value *, Value *> Cache;
        Value *Resolved =
            materializeLiveInInPlace(SI->getValueOperand(), B, Cache, UseDT,
                                     BadStore);
        if (!Resolved)
          Resolved =
              resolveBoundaryStoredValue(SI->getValueOperand(),
                                         BadStore->getParent(), B, UseDT);
        if (!Resolved || isa<UndefValue>(Resolved))
          return nullptr;
        if (auto *ResolvedI = dyn_cast<Instruction>(Resolved)) {
          if (!UseDT.dominates(ResolvedI, BadStore))
            return nullptr;
        }
        return Resolved;
      }

      auto *LI = dyn_cast<LoadInst>(Access);
      if (!LI)
        return nullptr;
      if (UseDT.dominates(LI, BadStore))
        return LI;
      return nullptr;
    };

    SmallPtrSet<BasicBlock *, 16> Visited;
    BasicBlock *Current = BadStore->getParent();
    Instruction *Before = BadStore;
    while (Current && Visited.insert(Current).second) {
      if (Value *Resolved = TryResolveInBlock(Current, Before))
        return Resolved;

      BasicBlock *SinglePred = Current->getSinglePredecessor();
      if (!SinglePred || !UseDT.dominates(SinglePred->getTerminator(), BadStore))
        break;
      Current = SinglePred;
      Before = nullptr;
    }

    if (DomTreeNode *Node = UseDT.getNode(BadStore->getParent())) {
      for (Node = Node->getIDom(); Node; Node = Node->getIDom()) {
        BasicBlock *DomBB = Node->getBlock();
        if (!DomBB || !Visited.insert(DomBB).second)
          continue;
        if (Value *Resolved = TryResolveInBlock(DomBB, nullptr))
          return Resolved;
      }
    }
    return nullptr;
  }

  bool repairReachableStageLocalCarrierSeeds(Function *F,
                                             DominatorTree *DT = nullptr) {
    if (!F || F->isDeclaration())
      return true;

    bool Ok = true;
    SmallVector<CarrierSeedStoreIssue, 8> Issues =
        collectReachableCarrierSeedStoreIssues(F);
    for (const CarrierSeedStoreIssue &Issue : Issues) {
      if (!Issue.BadStore || !Issue.CarrierAlloca)
        continue;

      IRBuilder<> B(Issue.BadStore);
      if (Value *Replacement =
              resolveStageLocalCarrierSeed(Issue.CarrierAlloca, Issue.BadStore,
                                          DT, B)) {
        errs() << "deepFission: repaired stage-local carrier seed in "
               << F->getName() << " for " << Issue.CarrierAlloca->getName()
               << " with " << *Replacement << "\n";
        Issue.BadStore->setOperand(0, Replacement);
        continue;
      }

      if (!Issue.ReachesLoadBeforeOverwrite) {
        errs() << "deepFission: erased dead stage-local carrier seed in "
               << F->getName() << " for " << Issue.CarrierAlloca->getName()
               << "\n";
        Issue.BadStore->eraseFromParent();
        continue;
      }

      errs() << "deepFission: failed to repair stage-local carrier seed in "
             << F->getName() << " for " << Issue.CarrierAlloca->getName()
             << " at " << *Issue.BadStore << "\n";
      if (Issue.NearestAccess)
        errs() << "deepFission:   nearest prior carrier access: "
               << *Issue.NearestAccess << "\n";
      Ok = false;
    }
    return Ok;
  }

  bool validateReachableCarrierSeedStores(Function *F) {
    if (!F || F->isDeclaration() || F->empty())
      return true;
    bool Ok = true;
    SmallVector<CarrierSeedStoreIssue, 8> Issues =
        collectReachableCarrierSeedStoreIssues(F);
    for (const CarrierSeedStoreIssue &Issue : Issues) {
      if (!Issue.ReachesLoadBeforeOverwrite)
        continue;
      errs() << "deepFission: reachable carrier seed store of undef in "
             << F->getName() << ": " << *Issue.BadStore << "\n";
      errs() << "deepFission:   carrier=" << Issue.CarrierAlloca->getName()
             << " block=" << Issue.BadStore->getParent()->getName() << "\n";
      if (Issue.NearestAccess)
        errs() << "deepFission:   nearest prior carrier access: "
               << *Issue.NearestAccess << "\n";
      Ok = false;
    }
    return Ok;
  }

  Function *splitFunction(LLVMContext &Context, Function *F,
                          Instruction *SyncInst, unsigned SplitIndex,
                          unsigned WorkingStageIndex,
                          unsigned &NextSpillIndex) {
    IRBuilder<> Builder(Context);
    if (!F || !SyncInst || SyncInst->getFunction() != F ||
        !isBarrierMarkerInst(SyncInst)) {
      errs() << "deepFission: skipped invalid sync marker in "
             << (F ? F->getName() : "<null>") << "\n";
      return nullptr;
    }

    dumpFunctionIR("splitFunction input", F);
    dumpInstructionIR("splitFunction chosen sync", SyncInst);

    auto &DT = getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
    auto &FunctionLI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
    Loop *BarrierLoop = FunctionLI.getLoopFor(SyncInst->getParent());

    BasicBlock *BB = SyncInst->getParent();
    BasicBlock *SplitPoint =
        BB->splitBasicBlock(SyncInst, "syncpoint." + std::to_string(SplitIndex));

    std::set<BasicBlock *> PostSplitRegion = collectPostSplitRegion(SplitPoint);
    std::set<BasicBlock *> EntryReachableWithoutSplit =
        collectReachableBlocksAvoiding(F, SplitPoint);
    std::set<BasicBlock *> BoundaryCutRegion = collectBoundaryCutRegion(
        SplitPoint, PostSplitRegion, EntryReachableWithoutSplit);
    std::vector<Value *> LiveIns = collectPostSplitLiveIns(F, PostSplitRegion);
    unsigned UnresolvedLiveIns = 0;
    SmallVector<Value *, 8> BoundaryLiveIns;
    for (Value *V : LiveIns) {
      if (isPostRegionLocalAllocaLiveIn(V, PostSplitRegion))
        continue;
      BoundaryLiveIns.push_back(V);
    }
    SmallVector<BoundaryValueInfo, 8> BoundaryValues = collectBoundaryValueInfos(
        F, BB->getTerminator(), BoundaryLiveIns, DT, UnresolvedLiveIns);
    if (UnresolvedLiveIns != 0) {
      errs() << "deepFission: skip unsafe split for " << F->getName()
             << " due to unresolved live-ins=" << UnresolvedLiveIns << "\n";
      return nullptr;
    }

    SmallVector<BasicBlock *, 8> BoundaryPreds =
        collectBoundaryPreds(F, BoundaryCutRegion, EntryReachableWithoutSplit);
    SmallVector<BoundaryEdgeValue, 8> BoundaryEdges =
        collectBoundaryEdges(F, BoundaryCutRegion, EntryReachableWithoutSplit);
    if (DeepFissionDumpIR) {
      errs() << "deepFission: boundary cut region for " << F->getName()
             << " = " << BoundaryCutRegion.size() << "\n";
      for (BasicBlock *CutBB : BoundaryCutRegion)
        errs() << "deepFission:   cut " << CutBB->getName() << "\n";
      errs() << "deepFission: boundary preds for " << F->getName() << " = "
             << BoundaryPreds.size() << "\n";
      for (BasicBlock *Pred : BoundaryPreds)
        errs() << "deepFission:   pred " << Pred->getName() << "\n";
      errs() << "deepFission: boundary edges for " << F->getName() << " = "
             << BoundaryEdges.size() << "\n";
      for (const BoundaryEdgeValue &Edge : BoundaryEdges) {
        errs() << "deepFission:   edge " << Edge.Pred->getName() << " -> "
               << Edge.OriginalSucc->getName() << "\n";
      }
    }
    if (BoundaryEdges.empty()) {
      errs() << "deepFission: skip unsafe split for " << F->getName()
             << " due to missing boundary edges\n";
      return nullptr;
    }

    bool PostRegionHadWork = false;
    for (BasicBlock *RBB : PostSplitRegion) {
      for (Instruction &RI : *RBB) {
        if (isa<DbgInfoIntrinsic>(&RI) || RI.isTerminator())
          continue;
        if (auto *II = dyn_cast<IntrinsicInst>(&RI)) {
          Intrinsic::ID ID = II->getIntrinsicID();
          if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end)
            continue;
        }
        if (isBarrierMarkerInst(&RI))
          continue;
        PostRegionHadWork = true;
        break;
      }
      if (PostRegionHadWork)
        break;
    }

    for (BoundaryValueInfo &Info : BoundaryValues) {
      if (Info.Kind == BoundaryValueKind::Rematerialize)
        continue;
      Info.SpillGlobal =
          createSpillGlobal(*F->getParent(), F, Info.StorageType,
                            NextSpillIndex++);
      errs() << "deepFission: created spill global "
             << Info.SpillGlobal->getName() << " for " << *Info.Value << "\n";
    }

    std::string Candidate =
        buildWorkingStageFunctionName(*F->getParent(), F, WorkingStageIndex);
    auto *NewFunc =
        Function::Create(F->getFunctionType(), F->getLinkage(), Candidate,
                         F->getParent());
    ValueToValueMapTy VMap;
    auto NewArgIt = NewFunc->arg_begin();
    for (Argument &Arg : F->args()) {
      NewArgIt->setName(Arg.getName());
      VMap[&Arg] = &(*NewArgIt++);
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(NewFunc, F, VMap, false, Returns);

    if (Instruction *MappedSync = dyn_cast<Instruction>(VMap[SyncInst]))
      MappedSync->eraseFromParent();

    bool ConsumerSatisfied = true;
    Instruction *EntryInsertPt = getStageEntryInsertPoint(NewFunc);
    if (!EntryInsertPt)
      return nullptr;
    IRBuilder<> EntryBuilder(EntryInsertPt);
    Value *EntryLinearTid = buildLinearThreadIndex(EntryBuilder, NewFunc);
    DenseMap<Value *, Value *> MaterializeCache;
    for (BoundaryValueInfo &Info : BoundaryValues) {
      auto MapIt = VMap.find(Info.Value);
      if (MapIt == VMap.end()) {
        errs() << "deepFission: missing mapped live-in in cloned stage for "
               << *Info.Value << "\n";
        ConsumerSatisfied = false;
        continue;
      }

      if (Info.Kind == BoundaryValueKind::Rematerialize) {
        Value *Mat = materializeLiveInInFunction(Info.Value, NewFunc, VMap,
                                                 EntryBuilder, MaterializeCache);
        auto *MappedI = dyn_cast<Instruction>(MapIt->second);
        if (!Mat || !MappedI) {
          errs() << "deepFission: failed to rematerialize live-in "
                 << *Info.Value << " in " << NewFunc->getName() << "\n";
          ConsumerSatisfied = false;
          continue;
        }
        if (MappedI != Mat)
          MappedI->replaceAllUsesWith(Mat);
        continue;
      }

      if (!Info.SpillGlobal || !EntryLinearTid) {
        errs() << "deepFission: failed to seed spill-backed live-in "
               << *Info.Value << " in " << NewFunc->getName() << "\n";
        ConsumerSatisfied = false;
        continue;
      }

      Value *Slot = buildSpillSlot(EntryBuilder, Info.SpillGlobal, EntryLinearTid);
      if (!Slot) {
        ConsumerSatisfied = false;
        continue;
      }

      if (Info.Kind == BoundaryValueKind::SpillSimpleAllocaSeed ||
          Info.Kind == BoundaryValueKind::SpillReg2MemCarrier) {
        auto *MappedAI = dyn_cast<AllocaInst>(MapIt->second);
        if (!MappedAI) {
          errs() << "deepFission: expected mapped alloca seed for "
                 << *Info.Value << " in " << NewFunc->getName() << "\n";
          ConsumerSatisfied = false;
          continue;
        }
        auto *Reload =
            EntryBuilder.CreateLoad(Info.StorageType, Slot, "df.spill.reload");
        StoreInst *SeedStore = EntryBuilder.CreateStore(Reload, MappedAI);
        SeedStore->setAlignment(MappedAI->getAlignment());
        continue;
      }

      auto *MappedI = dyn_cast<Instruction>(MapIt->second);
      if (!MappedI) {
        errs() << "deepFission: expected mapped instruction for spill live-in "
               << *Info.Value << " in " << NewFunc->getName() << "\n";
        ConsumerSatisfied = false;
        continue;
      }
      auto *Reload =
          EntryBuilder.CreateLoad(Info.StorageType, Slot, "df.spill.reload");
      MappedI->replaceAllUsesWith(Reload);
    }
    if (!ConsumerSatisfied)
      return nullptr;

    auto *MappedSplitTerm = dyn_cast<Instruction>(VMap[SplitPoint->getTerminator()]);
    if (!MappedSplitTerm)
      return nullptr;
    auto *NewSplitBB = MappedSplitTerm->getParent();

    std::set<BasicBlock *> NewPostSplitRegion;
    for (BasicBlock *OldBB : PostSplitRegion) {
      auto It = VMap.find(OldBB->getTerminator());
      if (It == VMap.end())
        continue;
      if (Instruction *MappedTerm = dyn_cast<Instruction>(It->second)) {
        if (BasicBlock *MappedBB = MappedTerm->getParent())
          NewPostSplitRegion.insert(MappedBB);
      }
    }

    if (BarrierLoop) {
      errs() << "deepFission: splitting loop-contained barrier in function "
             << F->getName() << " loop header="
             << BarrierLoop->getHeader()->getName() << "\n";
    }

    BoundaryExitPlan ExitPlan =
        buildBoundaryExitBlock(F, BoundaryValues, BoundaryEdges, DT);
    if (!ExitPlan.ExitBlock) {
      errs() << "deepFission: skip unsafe split for " << F->getName()
             << " due to boundary export failure\n";
      return nullptr;
    }
    if (DeepFissionDumpIR) {
      errs() << "deepFission: boundary export values for " << F->getName()
             << "\n";
      for (const BoundaryEdgeValue &Edge : ExitPlan.EdgeValues) {
        unsigned ExportIdx = 0;
        for (const BoundaryValueInfo &Info : BoundaryValues) {
          if (Info.Kind == BoundaryValueKind::Rematerialize)
            continue;
          if (ExportIdx >= Edge.ExportedValues.size())
            break;
          errs() << "deepFission:   edge " << Edge.Pred->getName() << " -> "
                 << Edge.OriginalSucc->getName() << " value=" << *Info.Value
                 << " exported as " << *Edge.ExportedValues[ExportIdx++] << "\n";
        }
      }
    }

    std::set<BasicBlock *> SplitEraseList;
    for (BasicBlock &CandBB : *NewFunc) {
      BasicBlock *Pred = &CandBB;
      if (Pred == &NewFunc->getEntryBlock())
        continue;
      if (!NewPostSplitRegion.count(Pred))
        SplitEraseList.insert(Pred);
    }

    sanitizeClonedEntryBlock(&NewFunc->getEntryBlock());
    Builder.SetInsertPoint(NewFunc->getEntryBlock().getTerminator());
    Builder.CreateBr(NewSplitBB);
    NewFunc->getEntryBlock().getTerminator()->eraseFromParent();

    for (BasicBlock *BBToTrim : SplitEraseList)
      replaceTerminatorWithReturnAndRepairPhis(BBToTrim, Builder);

    (void)simplifyDeadControlFlow(F);
    (void)simplifyDeadControlFlow(NewFunc);
    (void)pruneUnreachableBlocks(F);
    (void)pruneUnreachableBlocks(NewFunc);
    if (!repairProducerBoundaryCarrierSeeds(F, BoundaryValues))
      return nullptr;
    if (!repairConsumerCarrierSeeds(NewFunc, BoundaryValues, VMap))
      return nullptr;
    DominatorTree PostCleanupDT(*F);
    if (!repairReachableStageLocalCarrierSeeds(F, &PostCleanupDT))
      return nullptr;
    (void)repairNonDominatingAllocaLoads(NewFunc);
    DominatorTree NewFuncDT(*NewFunc);
    if (!repairReachableStageLocalCarrierSeeds(NewFunc, &NewFuncDT))
      return nullptr;
    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] materializing canonical resume entries for producer "
           << F->getName() << " and consumer " << NewFunc->getName() << "\n";
    if (!materializeCanonicalResumeEntries(F))
      return nullptr;
    if (!materializeCanonicalResumeEntries(NewFunc))
      return nullptr;
    (void)simplifyDeadControlFlow(F);
    (void)simplifyDeadControlFlow(NewFunc);
    (void)pruneUnreachableBlocks(F);
    (void)pruneUnreachableBlocks(NewFunc);
    (void)repairNonDominatingAllocaLoads(F);
    (void)repairNonDominatingAllocaLoads(NewFunc);
    DominatorTree CanonicalizedDT(*F);
    if (!repairReachableStageLocalCarrierSeeds(F, &CanonicalizedDT))
      return nullptr;
    DominatorTree CanonicalizedNewFuncDT(*NewFunc);
    if (!repairReachableStageLocalCarrierSeeds(NewFunc,
                                               &CanonicalizedNewFuncDT))
      return nullptr;
    if (!validateReachableCarrierSeedStores(NewFunc))
      return nullptr;

    if (PostRegionHadWork && !hasReachableSubstantiveWork(NewFunc)) {
      errs() << "deepFission: WARNING post-split result became trivial for "
             << NewFunc->getName() << " (source=" << F->getName() << ")\n";
    }

    if (!verifyFunctionOrReport("splitFunction original", F))
      return nullptr;
    if (!verifyFunctionOrReport("splitFunction new function", NewFunc))
      return nullptr;

    dumpFunctionIR("splitFunction transformed original", F);
    dumpFunctionIR("splitFunction new split function", NewFunc);

    return NewFunc;
  }

  BasicBlock *makeLoopPreheader(LoopInfo &LI, DominatorTree &DT, Loop *L) {
    BasicBlock *Header = L->getHeader();
    std::vector<BasicBlock *> EnteringBBs;
    for (pred_iterator PI = pred_begin(Header); PI != pred_end(Header); ++PI) {
      if (!L->contains(*PI))
        EnteringBBs.push_back(*PI);
    }

    BasicBlock *Preheader =
        SplitBlockPredecessors(Header, EnteringBBs, ".preheader");
    if (Loop *Parent = L->getParentLoop())
      Parent->addBasicBlockToLoop(Preheader, LI);
    DT.splitBlock(Preheader);
    return Preheader;
  }

  bool loopContainsEarlyReturn(Loop *L) {
    if (!L)
      return false;
    for (BasicBlock *BB : L->blocks()) {
      auto *RI = dyn_cast<ReturnInst>(BB->getTerminator());
      if (RI && !isSyntheticReturn(RI))
        return true;
    }
    return false;
  }

  void splitLoop(LLVMContext &Context, CallInst *I, unsigned CloneNum) {
    if (!I || !I->getParent() || CloneNum == 0)
      return;

    auto *F = I->getFunction();
    dumpInstructionIR("splitLoop seed call", I);
    dumpFunctionIR("splitLoop caller before cloning", F);
    auto &LI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
    Loop *ThreadLoop = LI.getLoopFor(I->getParent());
    if (!ThreadLoop) {
      errs() << "deepFission: skipping call not enclosed by loop: " << *I
             << "\n";
      return;
    }

    auto IsDynamicIndex = [](Value *V) -> bool {
      if (auto *CI = dyn_cast_or_null<ConstantInt>(V))
        return CI->getSExtValue() != 0;
      return V != nullptr;
    };

    unsigned ActiveThreadDims = 0;
    if (Function *Callee = I->getCalledFunction()) {
      Value *TidX = nullptr;
      Value *TidY = nullptr;
      Value *TidZ = nullptr;
      unsigned MaxArgs = std::min<unsigned>(I->arg_size(), Callee->arg_size());
      for (unsigned AI = 0; AI < MaxArgs; ++AI) {
        auto ArgIt = Callee->arg_begin();
        std::advance(ArgIt, AI);
        StringRef ArgName = ArgIt->getName();
        if (ArgName == "threadIdx.x")
          TidX = I->getArgOperand(AI);
        else if (ArgName == "threadIdx.y")
          TidY = I->getArgOperand(AI);
        else if (ArgName == "threadIdx.z")
          TidZ = I->getArgOperand(AI);
      }
      if (IsDynamicIndex(TidX))
        ActiveThreadDims++;
      if (IsDynamicIndex(TidY))
        ActiveThreadDims++;
      if (IsDynamicIndex(TidZ))
        ActiveThreadDims++;
    }

    for (unsigned Depth = 1; Depth < ActiveThreadDims; ++Depth) {
      Loop *Parent = ThreadLoop->getParentLoop();
      if (!Parent)
        break;
      ThreadLoop = Parent;
    }

    if (loopContainsEarlyReturn(ThreadLoop)) {
      errs() << "deepFission: skipping loop fission due to in-loop return: "
             << *I << "\n";
      return;
    }

    if (!ThreadLoop->getLoopPreheader())
      (void)makeLoopPreheader(LI, DT, ThreadLoop);
    if (!ThreadLoop->getLoopPreheader()) {
      errs() << "deepFission: failed to materialize loop preheader for " << *I
             << "\n";
      return;
    }

    BasicBlock *Entering =
        ThreadLoop->getLoopPreheader()->getSinglePredecessor();
    if (!Entering) {
      errs() << "deepFission: loop without single preheader predecessor: " << *I
             << "\n";
      return;
    }

    Loop *PrevLoop = ThreadLoop;
    for (unsigned Idx = 0; Idx < CloneNum; ++Idx) {
      ValueToValueMapTy VMap;
      SmallVector<BasicBlock *, 4> NewBlocks;
      Loop *NewLoop = cloneLoopWithPreheader(
          PrevLoop->getLoopPreheader(), &F->getEntryBlock(), ThreadLoop, VMap,
          ".clone" + std::to_string(Idx), &LI, &DT, NewBlocks);
      remapInstructionsInBlocks(NewBlocks, VMap);
      Entering->getTerminator()->replaceUsesOfWith(PrevLoop->getLoopPreheader(),
                                                   NewLoop->getLoopPreheader());
      NewLoop->getHeader()->getTerminator()->replaceUsesOfWith(
          ThreadLoop->getExitBlock(), PrevLoop->getLoopPreheader());

      auto MappedIt = VMap.find(I);
      if (MappedIt == VMap.end())
        continue;
      auto *MappedSeed = dyn_cast<Instruction>(MappedIt->second);
      if (!MappedSeed)
        continue;

      std::vector<CallInst *> ClonedCalls;
      for (Instruction *Scan = MappedSeed;
           Scan && Scan->getParent() == MappedSeed->getParent() &&
           ClonedCalls.size() < CloneNum + 1; Scan = Scan->getNextNode()) {
        if (auto *CI = dyn_cast<CallInst>(Scan))
          ClonedCalls.push_back(CI);
      }
      if (ClonedCalls.size() != CloneNum + 1)
        continue;

      for (unsigned J = 0; J < ClonedCalls.size(); ++J) {
        if (J + Idx != CloneNum - 1)
          ClonedCalls[J]->eraseFromParent();
      }

      PrevLoop = NewLoop;
    }

    std::vector<CallInst *> OriginalCalls;
    for (Instruction *Scan = I;
         Scan && Scan->getParent() == I->getParent() &&
         OriginalCalls.size() < CloneNum + 1; Scan = Scan->getNextNode()) {
      if (auto *CI = dyn_cast<CallInst>(Scan))
        OriginalCalls.push_back(CI);
    }
    if (OriginalCalls.size() != CloneNum + 1)
      return;
    for (unsigned J = 0; J < CloneNum; ++J)
      OriginalCalls[J]->eraseFromParent();

    dumpFunctionIR("splitLoop caller after cloning", F);
  }

  Function *cloneSeedToWorkingStage(Function *Seed) {
    if (!Seed || Seed->isDeclaration())
      return nullptr;

    Module *M = Seed->getParent();
    std::string Name = buildWorkingStageFunctionName(*M, Seed, 0);
    auto *Clone =
        Function::Create(Seed->getFunctionType(), GlobalValue::InternalLinkage,
                         Name, M);
    ValueToValueMapTy VMap;
    auto NewArgIt = Clone->arg_begin();
    for (Argument &Arg : Seed->args()) {
      NewArgIt->setName(Arg.getName());
      VMap[&Arg] = &(*NewArgIt++);
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(Clone, Seed, VMap, false, Returns);
    return Clone;
  }

  void collectReferencedGlobalsFromValue(Value *V,
                                         SmallPtrSetImpl<GlobalVariable *> &Out) {
    if (!V)
      return;
    if (auto *GV = dyn_cast<GlobalVariable>(V)) {
      Out.insert(GV);
      return;
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      for (Value *Op : CE->operands())
        collectReferencedGlobalsFromValue(Op, Out);
    }
  }

  SmallVector<GlobalVariable *, 8> collectReferencedSpillGlobals(Function *F) {
    SmallVector<GlobalVariable *, 8> Ordered;
    SmallPtrSet<GlobalVariable *, 8> Seen;
    if (!F || F->isDeclaration())
      return Ordered;

    std::string Prefix = getDeepFissionBaseName(F) + ".df.spill.";
    for (Instruction &I : instructions(F)) {
      SmallPtrSet<GlobalVariable *, 4> LocalGlobals;
      for (Value *Op : I.operands())
        collectReferencedGlobalsFromValue(Op, LocalGlobals);
      for (GlobalVariable *GV : LocalGlobals) {
        if (!GV)
          continue;
        if (!GV->getName().startswith(Prefix))
          continue;
        if (Seen.insert(GV).second)
          Ordered.push_back(GV);
      }
    }
    llvm::sort(Ordered, [](GlobalVariable *L, GlobalVariable *R) {
      return L->getName() < R->getName();
    });
    return Ordered;
  }

  SmallVector<GlobalVariable *, 4> collectReferencedSharedGlobals(Function *F) {
    SmallVector<GlobalVariable *, 4> Ordered;
    SmallPtrSet<GlobalVariable *, 4> Seen;
    if (!F || F->isDeclaration())
      return Ordered;

    std::string SpillPrefix = getDeepFissionBaseName(F) + ".df.spill.";
    for (Instruction &I : instructions(F)) {
      if (isa<DbgInfoIntrinsic>(&I))
        continue;
      SmallPtrSet<GlobalVariable *, 4> LocalGlobals;
      for (Value *Op : I.operands())
        collectReferencedGlobalsFromValue(Op, LocalGlobals);
      for (GlobalVariable *GV : LocalGlobals) {
        if (!GV || !GV->isThreadLocal() || !isa<ArrayType>(GV->getValueType()))
          continue;
        if (GV->getName().startswith(SpillPrefix))
          continue;
        if (Seen.insert(GV).second)
          Ordered.push_back(GV);
      }
    }
    llvm::sort(Ordered, [](GlobalVariable *L, GlobalVariable *R) {
      return L->getName() < R->getName();
    });
    return Ordered;
  }

  bool rewriteSpillGlobalSlots(Function *F,
                               DenseMap<GlobalVariable *, Argument *> &SpillArgs) {
    if (!F || F->isDeclaration() || SpillArgs.empty())
      return false;

    SmallVector<GetElementPtrInst *, 16> ToRewrite;
    for (Instruction &I : instructions(F)) {
      auto *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP)
        continue;
      auto *GV =
          dyn_cast<GlobalVariable>(GEP->getPointerOperand()->stripPointerCasts());
      if (!GV || !SpillArgs.count(GV))
        continue;
      ToRewrite.push_back(GEP);
    }

    bool Changed = false;
    for (GetElementPtrInst *GEP : ToRewrite) {
      auto *GV =
          cast<GlobalVariable>(GEP->getPointerOperand()->stripPointerCasts());
      Argument *SpillArg = SpillArgs[GV];
      auto *SpillArrayTy = dyn_cast<ArrayType>(GV->getValueType());
      if (!SpillArrayTy || GEP->getNumIndices() < 2)
        continue;

      auto IdxIt = GEP->idx_begin();
      Value *FirstIdx = *IdxIt++;
      Value *LinearIdx = *IdxIt;
      auto *FirstCI = dyn_cast<ConstantInt>(FirstIdx);
      if (!FirstCI || !FirstCI->isZero())
        continue;

      IRBuilder<> B(GEP);
      Value *Linear = LinearIdx;
      if (!Linear->getType()->isIntegerTy(64))
        Linear = B.CreateZExtOrTrunc(
            Linear, Type::getInt64Ty(F->getContext()), "df.spill.idx");
      Value *Replacement =
          B.CreateInBoundsGEP(SpillArrayTy->getElementType(), SpillArg, Linear,
                              "df.spill.slot");
      GEP->replaceAllUsesWith(Replacement);
      if (GEP->use_empty())
        GEP->eraseFromParent();
      Changed = true;
    }
    return Changed;
  }

  GlobalVariable *getCanonicalizableSharedGlobal(Constant *C) {
    auto IsCanonicalSharedArray = [](GlobalVariable *GV) -> bool {
      return GV && GV->isThreadLocal() && isa<ArrayType>(GV->getValueType());
    };

    if (auto *GV = dyn_cast_or_null<GlobalVariable>(C)) {
      if (IsCanonicalSharedArray(GV))
        return GV;
      return nullptr;
    }

    auto *CE = dyn_cast_or_null<ConstantExpr>(C);
    if (!CE)
      return nullptr;

    switch (CE->getOpcode()) {
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
      return getCanonicalizableSharedGlobal(
          dyn_cast<Constant>(CE->getOperand(0)));
    case Instruction::GetElementPtr: {
      auto *GEP = cast<GEPOperator>(CE);
      auto *GV = dyn_cast<GlobalVariable>(
          GEP->getPointerOperand()->stripPointerCasts());
      if (!IsCanonicalSharedArray(GV))
        return nullptr;
      for (User::op_iterator IdxIt = GEP->idx_begin(), IdxEnd = GEP->idx_end();
           IdxIt != IdxEnd; ++IdxIt) {
        auto *CI = dyn_cast<ConstantInt>(*IdxIt);
        if (!CI || !CI->isZero())
          return nullptr;
      }
      return GV;
    }
    default:
      return nullptr;
    }
  }

  bool referencesCanonicalizableSharedGlobal(Constant *C) {
    if (!C)
      return false;
    if (getCanonicalizableSharedGlobal(C))
      return true;
    auto *CE = dyn_cast<ConstantExpr>(C);
    if (!CE)
      return false;
    for (unsigned I = 0; I < CE->getNumOperands(); ++I) {
      if (referencesCanonicalizableSharedGlobal(
              dyn_cast<Constant>(CE->getOperand(I))))
        return true;
    }
    return false;
  }

  Value *getOrCreateSharedGlobalBase(Function *F, GlobalVariable *GV,
                                     DenseMap<GlobalVariable *, Value *> &Cache) {
    if (!F || !GV)
      return nullptr;
    auto It = Cache.find(GV);
    if (It != Cache.end())
      return It->second;

    auto *ArrTy = dyn_cast<ArrayType>(GV->getValueType());
    if (!ArrTy)
      return nullptr;
    Type *ElemTy = ArrTy->getElementType();
    auto *ZeroArrTy = ArrayType::get(ElemTy, 0);
    Instruction *InsertPt = getStageEntryInsertPoint(F);
    if (!InsertPt)
      return nullptr;

    IRBuilder<NoFolder> B(F->getContext());
    B.SetInsertPoint(InsertPt);
    Value *Cast =
        B.CreateBitCast(GV, ZeroArrTy->getPointerTo(GV->getAddressSpace()),
                        GV->getName() + ".df.shared.arr");
    Value *Base =
        B.CreateInBoundsGEP(ZeroArrTy, Cast,
                            {B.getInt64(0), B.getInt64(0)},
                            GV->getName() + ".sharedMem.gep");
    Cache[GV] = Base;
    return Base;
  }

  Value *materializeSharedConstantExpr(
      Function *F, Constant *C, Instruction *InsertBefore,
      DenseMap<GlobalVariable *, Value *> &BaseCache,
      DenseMap<Constant *, Value *> &ExprCache) {
    if (!F || !C || !InsertBefore)
      return nullptr;

    auto ValueDominatesUse = [&](Value *V) -> bool {
      if (!V)
        return false;
      if (isa<Argument>(V) || isa<Constant>(V) || isa<GlobalValue>(V))
        return true;
      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getFunction() != F)
        return false;
      if (I->getParent() == InsertBefore->getParent()) {
        if (I == InsertBefore)
          return true;
        for (Instruction &Cur : *I->getParent()) {
          if (&Cur == I)
            return true;
          if (&Cur == InsertBefore)
            return false;
        }
        return false;
      }
      DominatorTree DT(*F);
      return DT.dominates(I, InsertBefore);
    };

    auto CachedIt = ExprCache.find(C);
    if (CachedIt != ExprCache.end() &&
        ValueDominatesUse(CachedIt->second))
      return CachedIt->second;

    if (GlobalVariable *GV = getCanonicalizableSharedGlobal(C)) {
      Value *Base = getOrCreateSharedGlobalBase(F, GV, BaseCache);
      if (!Base)
        return nullptr;
      Value *Replacement = Base;
      if (Replacement->getType() != C->getType()) {
        IRBuilder<NoFolder> B(F->getContext());
        B.SetInsertPoint(InsertBefore);
        Replacement = B.CreateBitCast(
            Replacement, C->getType(), GV->getName() + ".df.shared.cast");
      }
      ExprCache[C] = Replacement;
      return Replacement;
    }

    auto *CE = dyn_cast<ConstantExpr>(C);
    if (!CE)
      return nullptr;

    SmallVector<Value *, 8> Operands;
    Operands.reserve(CE->getNumOperands());
    for (unsigned I = 0; I < CE->getNumOperands(); ++I) {
      Value *OperandV = CE->getOperand(I);
      if (auto *OperandC = dyn_cast<Constant>(OperandV)) {
        if (referencesCanonicalizableSharedGlobal(OperandC)) {
          OperandV = materializeSharedConstantExpr(F, OperandC, InsertBefore,
                                                   BaseCache, ExprCache);
          if (!OperandV)
            return nullptr;
        }
      }
      Operands.push_back(OperandV);
    }

    IRBuilder<NoFolder> B(F->getContext());
    B.SetInsertPoint(InsertBefore);
    Value *Result = nullptr;
    switch (CE->getOpcode()) {
    case Instruction::BitCast:
      Result = B.CreateBitCast(Operands[0], CE->getType(), "df.shared.bitcast");
      break;
    case Instruction::AddrSpaceCast:
      Result =
          B.CreateAddrSpaceCast(Operands[0], CE->getType(), "df.shared.asc");
      break;
    case Instruction::GetElementPtr: {
      auto *GEP = cast<GEPOperator>(CE);
      SmallVector<Value *, 8> Indices;
      for (unsigned I = 1; I < Operands.size(); ++I)
        Indices.push_back(Operands[I]);
      if (GEP->isInBounds())
        Result = B.CreateInBoundsGEP(GEP->getSourceElementType(), Operands[0],
                                     Indices, "df.shared.gep");
      else
        Result =
            B.CreateGEP(GEP->getSourceElementType(), Operands[0], Indices,
                        "df.shared.gep");
      break;
    }
    default:
      return nullptr;
    }

    ExprCache[C] = Result;
    return Result;
  }

  bool canonicalizeSharedGlobalBasesForCBE(Function *F) {
    if (!F || F->isDeclaration())
      return false;

    struct SharedUseRewrite {
      Instruction *User = nullptr;
      unsigned OperandNo = 0;
      Constant *Operand = nullptr;
      GlobalVariable *GV = nullptr;
    };

    SmallVector<SharedUseRewrite, 32> Rewrites;
    for (Instruction &I : instructions(F)) {
      if (isa<DbgInfoIntrinsic>(&I))
        continue;
      for (unsigned OperandNo = 0; OperandNo < I.getNumOperands(); ++OperandNo) {
        auto *C = dyn_cast<Constant>(I.getOperand(OperandNo));
        if (!C)
          continue;
        GlobalVariable *GV = getCanonicalizableSharedGlobal(C);
        if (!GV)
          continue;
        Rewrites.push_back({&I, OperandNo, C, GV});
      }
    }

    if (Rewrites.empty())
      return false;

    DenseMap<GlobalVariable *, Value *> BaseCache;
    DenseMap<Constant *, Value *> ExprCache;
    bool Changed = false;
    for (const SharedUseRewrite &Rewrite : Rewrites) {
      Value *Replacement = materializeSharedConstantExpr(
          F, Rewrite.Operand, Rewrite.User, BaseCache, ExprCache);
      if (!Replacement)
        continue;
      Rewrite.User->setOperand(Rewrite.OperandNo, Replacement);
      Changed = true;
    }
    return Changed;
  }

  bool rewriteSharedGlobalSlots(
      Function *F, DenseMap<GlobalVariable *, Argument *> &SharedArgs) {
    if (!F || F->isDeclaration() || SharedArgs.empty())
      return false;

    struct SharedUseRewrite {
      Instruction *User = nullptr;
      unsigned OperandNo = 0;
      Constant *Operand = nullptr;
    };

    SmallVector<SharedUseRewrite, 32> Rewrites;
    for (Instruction &I : instructions(F)) {
      if (isa<DbgInfoIntrinsic>(&I))
        continue;
      for (unsigned OperandNo = 0; OperandNo < I.getNumOperands(); ++OperandNo) {
        auto *C = dyn_cast<Constant>(I.getOperand(OperandNo));
        if (!C || !referencesCanonicalizableSharedGlobal(C))
          continue;
        Rewrites.push_back({&I, OperandNo, C});
      }
    }

    if (Rewrites.empty())
      return false;

    DenseMap<GlobalVariable *, Value *> BaseCache;
    for (auto &Entry : SharedArgs)
      BaseCache[Entry.first] = Entry.second;
    DenseMap<Constant *, Value *> ExprCache;
    bool Changed = false;
    for (const SharedUseRewrite &Rewrite : Rewrites) {
      Value *Replacement = materializeSharedConstantExpr(
          F, Rewrite.Operand, Rewrite.User, BaseCache, ExprCache);
      if (!Replacement)
        continue;
      Rewrite.User->setOperand(Rewrite.OperandNo, Replacement);
      Changed = true;
    }
    return Changed;
  }

  bool applyAuthorityLaneLowering(Function *F, Argument *ActiveMaskArg,
                                  const CooperativePhaseRiskReport &Report) {
    if (!F || F->isDeclaration() || !ActiveMaskArg)
      return false;

    BasicBlock *BodyBB = findActiveBodyEntryBlock(F);
    if (!BodyBB)
      return false;

    BasicBlock *AuthorityBodyBB = nullptr;
    Instruction *SplitPt = BodyBB->getFirstNonPHIOrDbgOrLifetime();
    if (SplitPt && SplitPt != BodyBB->getTerminator()) {
      AuthorityBodyBB =
          BodyBB->splitBasicBlock(SplitPt, "df.phase.authority.body");
    } else {
      auto *BI = dyn_cast<BranchInst>(BodyBB->getTerminator());
      if (!BI || !BI->isUnconditional())
        return false;
      AuthorityBodyBB = BI->getSuccessor(0);
    }
    if (!AuthorityBodyBB)
      return false;

    BasicBlock *CheckBB = BasicBlock::Create(F->getContext(),
                                             "df.phase.authority.check", F,
                                             AuthorityBodyBB);
    BasicBlock *SkipBB = BasicBlock::Create(F->getContext(),
                                            "df.phase.authority.skip", F,
                                            AuthorityBodyBB);

    IRBuilder<> B(BodyBB->getTerminator());
    auto FindCanonicalExitBlock = [&](Function *Fn) -> BasicBlock * {
      if (!Fn)
        return nullptr;
      for (BasicBlock &BB : *Fn) {
        if (BB.hasName() && BB.getName().startswith("df.exit"))
          return &BB;
      }
      return nullptr;
    };

    Value *AuthorityCond = nullptr;
    std::string PredicateSource;
    auto BuildAxisPredicate = [&](IRBuilder<> &IB, StringRef AxisName,
                                  StringRef BlockDimName, StringRef BlockIdxName,
                                  uint64_t ModuloDivisor) -> Value * {
      Argument *AxisArg = findNamedArgument(F, AxisName);
      if (!AxisArg || !AxisArg->getType()->isIntegerTy())
        return nullptr;

      auto *I64Ty = Type::getInt64Ty(F->getContext());
      auto ToI64 = [&](Value *V) -> Value * {
        if (!V || !V->getType()->isIntegerTy())
          return nullptr;
        if (V->getType()->isIntegerTy(64))
          return V;
        return IB.CreateZExtOrTrunc(V, I64Ty, "df.phase.axis.ext");
      };

      Value *AxisCoord = ToI64(AxisArg);
      if (!AxisCoord)
        return nullptr;

      if (ModuloDivisor > 1) {
        Value *BlockDim = ToI64(findNamedArgument(F, BlockDimName));
        Value *BlockIdx = ToI64(findNamedArgument(F, BlockIdxName));
        if (BlockDim && BlockIdx) {
          Value *BlockOffset = IB.CreateMul(BlockDim, BlockIdx,
                                            "df.phase.axis.blockoff");
          AxisCoord = IB.CreateAdd(AxisCoord, BlockOffset,
                                   "df.phase.axis.global");
          Value *AxisModuloValue = IB.CreateURem(
              AxisCoord, ConstantInt::get(I64Ty, ModuloDivisor),
              "df.phase.axis.mod");
          PredicateSource = "axis_modulo";
          return IB.CreateICmpEQ(AxisModuloValue, ConstantInt::get(I64Ty, 0),
                                 "df.phase.authority");
        }
      }

      PredicateSource = "axis_eq_zero";
      return IB.CreateICmpEQ(AxisCoord, ConstantInt::get(I64Ty, 0),
                             "df.phase.authority");
    };

    // Always go through explicit authority-check block so final C lowering
    // materializes an early non-authority return shape.
    B.CreateBr(CheckBB);
    BodyBB->getTerminator()->eraseFromParent();

    IRBuilder<> CheckBuilder(CheckBB);
    if (Report.Axis == AuthorityAxisKind::ThreadIdxY) {
      AuthorityCond = BuildAxisPredicate(CheckBuilder, "threadIdx.y",
                                         "blockDim.y", "blockIdx.y",
                                         Report.AxisModulo);
    } else if (Report.Axis == AuthorityAxisKind::ThreadIdxZ) {
      AuthorityCond = BuildAxisPredicate(CheckBuilder, "threadIdx.z",
                                         "blockDim.z", "blockIdx.z",
                                         Report.AxisModulo);
    }

    if (!AuthorityCond) {
      Value *LinearTid = buildLinearThreadIndex(CheckBuilder, F);
      if (!LinearTid)
        return false;
      PredicateSource = "linear_tid_eq_zero";
      AuthorityCond = CheckBuilder.CreateICmpEQ(
          LinearTid, ConstantInt::get(Type::getInt64Ty(F->getContext()), 0),
          "df.phase.authority");
    }

    errs() << "deepFission: [DF][PHASE] event=authority.select func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " axis=" << authorityAxisKindName(Report.Axis)
           << " axis_modulo=" << Report.AxisModulo
           << " source=" << PredicateSource << "\n";

    CheckBuilder.CreateCondBr(AuthorityCond, AuthorityBodyBB, SkipBB);
    errs() << "deepFission: [DF][PHASE] event=authority.early_return func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " axis=" << authorityAxisKindName(Report.Axis)
           << " axis_modulo=" << Report.AxisModulo
           << " source=" << PredicateSource << "\n";

    IRBuilder<> SkipBuilder(SkipBB);
    BasicBlock *ExitBB = FindCanonicalExitBlock(F);
    if (ExitBB) {
      // Keep final-stage return shape canonical: route non-authority lanes
      // through the existing df.exit root instead of introducing a new return.
      SkipBuilder.CreateBr(ExitBB);
    } else {
      ReturnInst *RI = nullptr;
      if (F->getReturnType()->isVoidTy())
        RI = SkipBuilder.CreateRetVoid();
      else
        RI = SkipBuilder.CreateRet(UndefValue::get(F->getReturnType()));
      markSyntheticReturn(RI, true);
    }

    markFunctionPhaseFlag(F, "tulip.df.phase.authority.early_return");

    errs() << "deepFission: [DF][PHASE] event=authority.apply func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " body=" << BodyBB->getName()
           << " check=" << CheckBB->getName()
           << " authority_body=" << AuthorityBodyBB->getName()
           << " skip=" << SkipBB->getName()
           << " exit=" << (ExitBB ? ExitBB->getName() : StringRef("<ret>"))
           << "\n";
    return true;
  }

  bool installActiveMaskGuard(Function *F, Argument *ActiveMaskArg) {
    if (!F || F->isDeclaration() || !ActiveMaskArg)
      return false;

    BasicBlock *EntryBB = &F->getEntryBlock();
    Instruction *SplitPt = EntryBB->getFirstNonPHIOrDbgOrLifetime();
    while (SplitPt && isa<AllocaInst>(SplitPt))
      SplitPt = SplitPt->getNextNode();
    BasicBlock *BodyBB = nullptr;
    if (!SplitPt || SplitPt == EntryBB->getTerminator()) {
      auto *BI = dyn_cast<BranchInst>(EntryBB->getTerminator());
      if (!BI || !BI->isUnconditional())
        return false;
      BodyBB = BI->getSuccessor(0);
    } else {
      BodyBB = EntryBB->splitBasicBlock(SplitPt, "df.active.body");
    }

    BasicBlock *SkipBB = BasicBlock::Create(F->getContext(), "df.active.skip", F,
                                            BodyBB);
    IRBuilder<> B(EntryBB->getTerminator());
    Value *LinearTid = buildLinearThreadIndex(B, F);
    if (!LinearTid)
      return false;
    Value *Slot =
        B.CreateInBoundsGEP(Type::getInt1Ty(F->getContext()), ActiveMaskArg,
                            LinearTid, "df.active.slot");
    Value *Active = B.CreateLoad(Type::getInt1Ty(F->getContext()), Slot,
                                 "df.active");
    B.CreateCondBr(Active, BodyBB, SkipBB);
    EntryBB->getTerminator()->eraseFromParent();

    IRBuilder<> SkipBuilder(SkipBB);
    ReturnInst *RI = nullptr;
    if (F->getReturnType()->isVoidTy())
      RI = SkipBuilder.CreateRetVoid();
    else
      RI = SkipBuilder.CreateRet(UndefValue::get(F->getReturnType()));
    markSyntheticReturn(RI);
    return true;
  }

  bool lowerEarlyReturnsToActiveMask(
      Function *F, Argument *ActiveMaskArg,
      ArrayRef<ExitSemanticInfo> ExplicitExitSemantics = {}) {
    if (!F || F->isDeclaration() || !ActiveMaskArg)
      return false;

    PostDominatorTree PDT(*F);
    BasicBlock *EntryBB = &F->getEntryBlock();
    BasicBlock *StageBodyBB = nullptr;
    if (auto *EntryBr = dyn_cast<BranchInst>(EntryBB->getTerminator())) {
      if (EntryBr->isConditional()) {
        BasicBlock *Succ0 = EntryBr->getSuccessor(0);
        BasicBlock *Succ1 = EntryBr->getSuccessor(1);
        auto IsSkipBlock = [](BasicBlock *BB) {
          return BB && BB->hasName() &&
                 BB->getName().startswith("df.active.skip");
        };
        if (IsSkipBlock(Succ0) && !IsSkipBlock(Succ1))
          StageBodyBB = Succ1;
        else if (IsSkipBlock(Succ1) && !IsSkipBlock(Succ0))
          StageBodyBB = Succ0;
      }
    }

    auto IsLaneTerminatingReturn = [&](ReturnInst *RI) {
      if (!RI)
        return false;
      for (const ExitSemanticInfo &Info : ExplicitExitSemantics) {
        if (Info.RootReturn != RI)
          continue;
        return Info.Kind == ExitSemanticKind::LaneTerminating;
      }
      if (isLaneTerminatingReturnMarked(RI))
        return true;
      if (isSyntheticReturn(RI) && syntheticReturnPreservesActiveMask(RI))
        return false;
      if (!StageBodyBB)
        return !isSyntheticReturn(RI) ||
               !syntheticReturnPreservesActiveMask(RI);
      return !PDT.dominates(RI->getParent(), StageBodyBB);
    };

    SmallVector<ReturnInst *, 8> Returns;
    SmallPtrSet<ReturnInst *, 8> ExplicitPreserveActiveRoots;
    for (Instruction &I : instructions(F)) {
      if (auto *RI = dyn_cast<ReturnInst>(&I)) {
        Returns.push_back(RI);
      }
    }
    for (const ExitSemanticInfo &Info : ExplicitExitSemantics) {
      if (Info.Kind == ExitSemanticKind::PreserveActive && Info.RootReturn)
        ExplicitPreserveActiveRoots.insert(Info.RootReturn);
    }

    errs() << "deepFission: lower exits func=" << F->getName()
           << " explicit=" << ExplicitExitSemantics.size()
           << " returns=" << Returns.size() << "\n";
    errs() << "deepFission: [DF][EXIT] event=lower.start func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " explicit=" << ExplicitExitSemantics.size()
           << " returns=" << Returns.size() << "\n";
    for (const ExitSemanticInfo &Info : ExplicitExitSemantics) {
      errs() << "deepFission:   explicit kind="
             << (Info.Kind == ExitSemanticKind::LaneTerminating
                     ? "lane_terminating"
                     : "preserve_active")
             << " root="
             << (Info.RootReturn ? Info.RootReturn->getParent()->getName()
                                 : StringRef("<null>"))
             << " entry_edges=" << Info.EntryEdges.size() << "\n";
      errs() << "deepFission: [DF][EXIT] event=lower.explicit func="
             << F->getName() << " stage=" << getStageIndexForLog(F)
             << " kind=" << exitSemanticKindName(Info.Kind) << " root="
             << (Info.RootReturn ? Info.RootReturn->getParent()->getName()
                                 : StringRef("<null>"))
             << " entry_edges=" << Info.EntryEdges.size() << "\n";
    }

    if (Returns.empty() && ExplicitExitSemantics.empty())
      return false;

    auto SplitLaneTerminatingExitEdge = [&](BranchInst *BI, unsigned SuccIdx,
                                            BasicBlock *RetBB) -> bool {
      if (!BI || !RetBB)
        return false;
      if (RetBB->hasName() && RetBB->getName().startswith("df.active.exit"))
        return false;
      BasicBlock *PredBB = BI->getParent();
      if (!PredBB)
        return false;

      BasicBlock *EdgeBB =
          BasicBlock::Create(F->getContext(), "df.active.exit", F, RetBB);
      for (Instruction &I : *RetBB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break;
        int IncomingIdx = PN->getBasicBlockIndex(PredBB);
        if (IncomingIdx >= 0)
          PN->setIncomingBlock(IncomingIdx, EdgeBB);
      }

      IRBuilder<> EB(EdgeBB);
      Value *LinearTid = buildLinearThreadIndex(EB, F);
      if (LinearTid) {
        Value *Slot = EB.CreateInBoundsGEP(Type::getInt1Ty(F->getContext()),
                                          ActiveMaskArg, LinearTid,
                                          "df.active.slot");
        EB.CreateStore(ConstantInt::getFalse(Type::getInt1Ty(F->getContext())),
                       Slot);
      }
      EB.CreateBr(RetBB);
      BI->setSuccessor(SuccIdx, EdgeBB);
      return true;
    };

    bool Changed = false;
    SmallPtrSet<ReturnInst *, 8> RootsCoveredByEdgeSplit;
    for (const ExitSemanticInfo &Info : ExplicitExitSemantics) {
      if (Info.Kind != ExitSemanticKind::LaneTerminating)
        continue;
      for (const ExitSemanticEdge &Edge : Info.EntryEdges) {
        BranchInst *BI = Edge.Branch;
        unsigned SuccIdx = Edge.SuccessorIndex;
        if (!BI || !BI->isConditional() || SuccIdx >= BI->getNumSuccessors())
          continue;
        BasicBlock *RetBB = BI->getSuccessor(SuccIdx);
        if (!RetBB)
          continue;
        if (SplitLaneTerminatingExitEdge(BI, SuccIdx, RetBB)) {
          errs() << "deepFission:   split explicit lane exit edge func="
                 << F->getName() << " pred=" << BI->getParent()->getName()
                 << " succ=" << SuccIdx << " target=" << RetBB->getName()
                 << "\n";
          errs() << "deepFission: [DF][EXIT] event=lower.split_edge func="
                 << F->getName() << " stage=" << getStageIndexForLog(F)
                 << " pred=" << BI->getParent()->getName()
                 << " succ=" << SuccIdx << " target=" << RetBB->getName()
                 << "\n";
          if (Info.RootReturn)
            RootsCoveredByEdgeSplit.insert(Info.RootReturn);
          Changed = true;
        }
      }
    }

    BasicBlock *ExitBB = BasicBlock::Create(F->getContext(), "df.exit", F);
    IRBuilder<> ExitBuilder(ExitBB);
    if (F->getReturnType()->isVoidTy())
      ExitBuilder.CreateRetVoid();
    else
      ExitBuilder.CreateRet(UndefValue::get(F->getReturnType()));

    for (ReturnInst *RI : Returns) {
      if (!RI || RI->getParent() == ExitBB)
        continue;
      IRBuilder<> B(RI);
      bool PreserveActive = ExplicitPreserveActiveRoots.count(RI);
      bool CoveredByEdgeSplit = RootsCoveredByEdgeSplit.count(RI);
      if (!PreserveActive && IsLaneTerminatingReturn(RI) &&
          !CoveredByEdgeSplit) {
        Value *LinearTid = buildLinearThreadIndex(B, F);
        if (LinearTid) {
          Value *Slot = B.CreateInBoundsGEP(Type::getInt1Ty(F->getContext()),
                                           ActiveMaskArg, LinearTid,
                                           "df.active.slot");
          B.CreateStore(ConstantInt::getFalse(Type::getInt1Ty(F->getContext())),
                        Slot);
        }
      }
      B.CreateBr(ExitBB);
      RI->eraseFromParent();
      Changed = true;
    }
    errs() << "deepFission: [DF][EXIT] event=lower.done func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " changed=" << (Changed ? 1 : 0) << "\n";
    return Changed;
  }

  bool rewriteCarryPreloadToMicroKernels(
      Function *F, CooperativePhaseRiskReport &Report) {
    if (!F || F->isDeclaration())
      return false;

    errs() << "deepFission: [DF][PHASE] event=subregion.rewrite.start func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " carry_preload=" << (Report.HasCarryPreloadSubregion ? 1 : 0)
           << "\n";

    if (!Report.HasCarryPreloadSubregion) {
      Report.HasMicroKernelRewrite = false;
      errs() << "deepFission: [DF][PHASE] event=subregion.rewrite.done func="
             << F->getName() << " stage=" << getStageIndexForLog(F)
             << " applied=0 reason=no_subregion\n";
      return true;
    }

    // This path is intentionally strict: if the stage still has a carry/preload
    // phase pattern and we cannot prove a canonical micro-kernel rewrite,
    // final-stage materialization must fail unless legacy compatibility is
    // explicitly enabled.
    Report.HasMicroKernelRewrite = false;
    Report.ResidualRiskReason = "carry_preload_rewrite_failed";
    errs() << "deepFission: [DF][PHASE] event=subregion.kernel.emit func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " kind=matvec_sub_block5x5 status=not_emitted\n";
    errs() << "deepFission: [DF][PHASE] event=subregion.kernel.emit func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " kind=matmul_sub_block5x5 status=not_emitted\n";
    errs() << "deepFission: [DF][PHASE] event=subregion.rewrite.done func="
           << F->getName() << " stage=" << getStageIndexForLog(F)
           << " applied=0 reason=pattern_not_yet_rewritten\n";
    return false;
  }

  bool fanoutAuthoritySpillWrites(Function *F, unsigned SpillLocalIdx,
                                  AuthorityAxisKind Axis, uint64_t AxisModulo,
                                  unsigned &WriteCount) {
    WriteCount = 0;
    if (!F || F->isDeclaration())
      return false;
    if (Axis == AuthorityAxisKind::None || AxisModulo <= 1)
      return true;

    std::string ArgName = "spill." + std::to_string(SpillLocalIdx);
    Argument *SpillArg = findNamedArgument(F, ArgName);
    if (!SpillArg)
      return false;

    StringRef AxisArg = authorityAxisArgumentName(Axis);
    if (AxisArg.empty())
      return false;

    auto *I64Ty = Type::getInt64Ty(F->getContext());
    auto ToI64 = [&](IRBuilder<> &B, Value *V, const Twine &Name) -> Value * {
      if (!V || !V->getType()->isIntegerTy())
        return nullptr;
      if (V->getType()->isIntegerTy(64))
        return V;
      return B.CreateZExtOrTrunc(V, I64Ty, Name);
    };

    auto BuildStride = [&](IRBuilder<> &B) -> Value * {
      Argument *BlockDimX = findNamedArgument(F, "blockDim.x");
      Value *StrideX = ToI64(B, BlockDimX, "df.phase.fanout.stride.x");
      if (!StrideX)
        return nullptr;
      if (Axis == AuthorityAxisKind::ThreadIdxY)
        return StrideX;
      if (Axis == AuthorityAxisKind::ThreadIdxZ) {
        Argument *BlockDimY = findNamedArgument(F, "blockDim.y");
        Value *StrideY = ToI64(B, BlockDimY, "df.phase.fanout.stride.y");
        if (!StrideY)
          return nullptr;
        return B.CreateMul(StrideX, StrideY, "df.phase.fanout.stride");
      }
      return nullptr;
    };

    SmallVector<StoreInst *, 8> CandidateStores;
    for (Instruction &I : instructions(F)) {
      auto *SI = dyn_cast<StoreInst>(&I);
      if (!SI)
        continue;
      if (getPointerRoot(SI->getPointerOperand()) != SpillArg)
        continue;
      if (!valueDependsOnNamedArgument(SI->getPointerOperand(), AxisArg))
        continue;
      CandidateStores.push_back(SI);
    }

    bool Changed = false;
    for (StoreInst *SI : CandidateStores) {
      Value *Ptr = SI->getPointerOperand()->stripPointerCasts();
      auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
      if (!GEP || GEP->getPointerOperand()->stripPointerCasts() != SpillArg ||
          GEP->getNumIndices() == 0)
        continue;

      Instruction *InsertPt = SI->getNextNode();
      IRBuilder<> B(InsertPt ? InsertPt : SI->getParent()->getTerminator());
      Value *Stride = BuildStride(B);
      if (!Stride)
        return false;
      Value *BaseIdx = ToI64(B, GEP->getOperand(GEP->getNumOperands() - 1),
                             "df.phase.fanout.baseidx");
      if (!BaseIdx)
        return false;

      for (uint64_t Lane = 1; Lane < AxisModulo; ++Lane) {
        Value *LaneStep = ConstantInt::get(I64Ty, Lane);
        Value *Offset = B.CreateMul(Stride, LaneStep, "df.phase.fanout.off");
        Value *FanoutIdx =
            B.CreateAdd(BaseIdx, Offset, "df.phase.fanout.idx");
        Value *Slot = B.CreateInBoundsGEP(
            SpillArg->getType()->getPointerElementType(), SpillArg, FanoutIdx,
            "df.phase.fanout.slot");
        B.CreateStore(SI->getValueOperand(), Slot);
        ++WriteCount;
      }
      Changed = true;
    }

    if (Changed) {
      markFunctionPhaseFlag(F, "tulip.df.phase.spill.fanout");
      errs() << "deepFission: [DF][PHASE] event=spill.fanout func="
             << F->getName() << " stage=" << getStageIndexForLog(F)
             << " spill_local_index=" << SpillLocalIdx
             << " axis=" << authorityAxisKindName(Axis)
             << " axis_modulo=" << AxisModulo
             << " write_count=" << WriteCount << "\n";
    }
    return true;
  }

  Function *createFinalStageFunction(Module &M, Function *Seed,
                                     FinalStageInfo &Info, unsigned StageIndex,
                                     const std::vector<FinalStageInfo> &AllStages) {
    if (!Seed || !Info.WorkingStage)
      return nullptr;

    std::vector<Type *> ParamTypes;
    for (Argument &Arg : Seed->args())
      ParamTypes.push_back(Arg.getType());
    for (GlobalVariable *GV : Info.UsedSpills) {
      auto *ArrTy = dyn_cast<ArrayType>(GV->getValueType());
      if (!ArrTy)
        return nullptr;
      ParamTypes.push_back(ArrTy->getElementType()->getPointerTo());
    }
    for (GlobalVariable *GV : Info.UsedSharedGlobals) {
      auto *ArrTy = dyn_cast<ArrayType>(GV->getValueType());
      if (!ArrTy)
        return nullptr;
      ParamTypes.push_back(ArrTy->getElementType()->getPointerTo());
    }
    ParamTypes.push_back(Type::getInt1PtrTy(M.getContext()));

    FunctionType *StageTy =
        FunctionType::get(Seed->getReturnType(), ParamTypes, Seed->isVarArg());
    std::string Name = buildStageFunctionName(M, Seed, StageIndex);
    auto *StageFunc = Function::Create(StageTy, GlobalValue::InternalLinkage,
                                       Name, &M);

    ValueToValueMapTy VMap;
    auto NewArgIt = StageFunc->arg_begin();
    for (Argument &OldArg : Info.WorkingStage->args()) {
      NewArgIt->setName(OldArg.getName());
      VMap[&OldArg] = &(*NewArgIt++);
    }

    DenseMap<GlobalVariable *, Argument *> SpillArgMap;
    for (unsigned I = 0; I < Info.UsedSpills.size(); ++I) {
      NewArgIt->setName("spill." + std::to_string(I));
      SpillArgMap[Info.UsedSpills[I]] = &(*NewArgIt++);
    }
    DenseMap<GlobalVariable *, Argument *> SharedArgMap;
    for (unsigned I = 0; I < Info.UsedSharedGlobals.size(); ++I) {
      NewArgIt->setName("shared." + std::to_string(I));
      SharedArgMap[Info.UsedSharedGlobals[I]] = &(*NewArgIt++);
    }
    Argument *ActiveMaskArg = &(*NewArgIt++);
    ActiveMaskArg->setName("active_mask");

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(StageFunc, Info.WorkingStage, VMap, false, Returns);
    SmallVector<ExitSemanticInfo, 4> RemappedExitSemantics;
    if (!remapExitSemantics(Info.ExitSemantics, VMap, RemappedExitSemantics))
      return nullptr;
    std::optional<Stage0EntrySemanticInfo> RemappedStage0Entry =
        remapStage0EntrySemanticInfo(Info.Stage0EntrySemantics, VMap);
    if (Info.Stage0EntrySemantics && !RemappedStage0Entry)
      return nullptr;
    Info.ExitSemantics = std::move(RemappedExitSemantics);
    errs() << "deepFission: [DF][FINAL] event=final_stage.remap_exits stage_func="
           << StageFunc->getName() << " stage=" << StageIndex
           << " count=" << Info.ExitSemantics.size() << "\n";
    for (const ExitSemanticInfo &ExitInfo : Info.ExitSemantics) {
      if (ExitInfo.Kind != ExitSemanticKind::PreserveActive)
        continue;
      if (!ExitInfo.RootReturn || !ExitInfo.RootReturn->getParent() ||
          ExitInfo.RootReturn->getParent()->getParent() != StageFunc) {
        errs() << "deepFission: [DF][FINAL] event=final_stage.remap_exits.error stage_func="
               << StageFunc->getName() << " stage=" << StageIndex
               << " reason=preserve_active_root_not_preserved\n";
        return nullptr;
      }
      errs() << "deepFission: [DF][FINAL] event=final_stage.preserve_root stage_func="
             << StageFunc->getName() << " stage=" << StageIndex << " root="
             << ExitInfo.RootReturn->getParent()->getName()
             << " entry_edges=" << ExitInfo.EntryEdges.size() << "\n";
    }
    Info.Stage0EntrySemantics = std::move(RemappedStage0Entry);
    (void)rewriteSpillGlobalSlots(StageFunc, SpillArgMap);
    (void)rewriteSharedGlobalSlots(StageFunc, SharedArgMap);
    bool RewroteBacksolveRecurrence =
        rewriteBacksolveRecurrenceToAuthorityScalar(StageFunc, ActiveMaskArg);
    if (!RewroteBacksolveRecurrence) {
      (void)installActiveMaskGuard(StageFunc, ActiveMaskArg);
    }
    (void)lowerEarlyReturnsToActiveMask(StageFunc, ActiveMaskArg,
                                        Info.ExitSemantics);

    CooperativePhaseRiskReport PhaseReport =
        analyzeCooperativePhaseRisk(StageFunc);
    if (PhaseReport.HasRisk) {
      if (!applyAuthorityLaneLowering(StageFunc, ActiveMaskArg, PhaseReport)) {
        errs() << "deepFission: [DF][PHASE] event=reject func="
               << StageFunc->getName() << " stage=" << StageIndex
               << " reason=authority_lowering_failed\n";
        return nullptr;
      }

      PhaseReport = analyzeCooperativePhaseRisk(StageFunc);

      bool RewriteOK = rewriteCarryPreloadToMicroKernels(StageFunc, PhaseReport);
      auto IsCarryPreloadRewriteResidual = [](const std::string &R) {
        return R == "carry_preload_rewrite_missing" ||
               R == "carry_preload_rewrite_failed";
      };
      bool AllowAuthorityOnlyFallback =
          DeepFissionAllowLegacyAuthorityOnly ||
          (PhaseReport.GuardedByAuthorityLane &&
           IsCarryPreloadRewriteResidual(PhaseReport.ResidualRiskReason));
      if (PhaseReport.HasCarryPreloadSubregion && !RewriteOK) {
        if (!AllowAuthorityOnlyFallback) {
          errs() << "deepFission: [DF][PHASE] event=reject func="
                 << StageFunc->getName() << " stage=" << StageIndex
                 << " reason=carry_preload_rewrite_failed\n";
          return nullptr;
        }
        errs() << "deepFission: [DF][PHASE] event=compat.legacy_authority_path func="
               << StageFunc->getName() << " stage=" << StageIndex
               << " reason=carry_preload_rewrite_failed\n";
      } else if (RewriteOK && PhaseReport.HasCarryPreloadSubregion) {
        markFunctionPhaseFlag(StageFunc, "tulip.df.phase.microkernel.rewrite");
      }

      bool NeedsSpillFanout = false;
      bool AppliedSpillFanout = false;
      for (unsigned SpillLocalIdx = 0; SpillLocalIdx < Info.UsedSpills.size();
           ++SpillLocalIdx) {
        GlobalVariable *GV = Info.UsedSpills[SpillLocalIdx];
        if (!detectDownstreamLaneIndexedSpillUse(AllStages, StageIndex, GV,
                                                 PhaseReport.Axis))
          continue;
        NeedsSpillFanout = true;
        markFunctionPhaseFlag(StageFunc, "tulip.df.phase.spill.fanout.required");
        unsigned FanoutWrites = 0;
        if (!fanoutAuthoritySpillWrites(StageFunc, SpillLocalIdx, PhaseReport.Axis,
                                        PhaseReport.AxisModulo, FanoutWrites)) {
          if (!DeepFissionAllowLegacyAuthorityOnly) {
            errs() << "deepFission: [DF][PHASE] event=reject func="
                   << StageFunc->getName() << " stage=" << StageIndex
                   << " reason=spill_fanout_missing spill_local_index="
                   << SpillLocalIdx << "\n";
            return nullptr;
          }
          errs() << "deepFission: [DF][PHASE] event=compat.legacy_authority_path func="
                 << StageFunc->getName() << " stage=" << StageIndex
                 << " reason=spill_fanout_missing spill_local_index="
                 << SpillLocalIdx << "\n";
          continue;
        }
        if (FanoutWrites > 0)
          AppliedSpillFanout = true;
      }
      if (NeedsSpillFanout && !AppliedSpillFanout &&
          !DeepFissionAllowLegacyAuthorityOnly) {
        errs() << "deepFission: [DF][PHASE] event=reject func="
               << StageFunc->getName() << " stage=" << StageIndex
               << " reason=spill_fanout_missing\n";
        return nullptr;
      }

      (void)simplifyDeadControlFlow(StageFunc);
      (void)pruneUnreachableBlocks(StageFunc);
      (void)repairNonDominatingAllocaLoads(StageFunc);
      CooperativePhaseRiskReport PostPhaseReport =
          analyzeCooperativePhaseRisk(StageFunc);
      errs() << "deepFission: [DF][PHASE] event=postcheck.summary func="
             << StageFunc->getName() << " stage=" << StageIndex
             << " applied=1"
             << " risk=" << (PostPhaseReport.HasRisk ? 1 : 0)
             << " guarded="
             << (PostPhaseReport.GuardedByAuthorityLane ? 1 : 0)
             << " early_authority_shape="
             << (PostPhaseReport.HasEarlyAuthorityReturnShape ? 1 : 0)
             << " subregion_rewritten="
             << (PostPhaseReport.HasMicroKernelRewrite ? 1 : 0)
             << " spill_fanout=" << (PostPhaseReport.HasSpillFanout ? 1 : 0)
             << " residual_reason="
             << (PostPhaseReport.ResidualRiskReason.empty()
                     ? StringRef("none")
                     : StringRef(PostPhaseReport.ResidualRiskReason))
             << " axis=" << authorityAxisKindName(PostPhaseReport.Axis)
             << " axis_modulo=" << PostPhaseReport.AxisModulo << "\n";
      if (PostPhaseReport.HasRisk) {
        bool AllowAuthorityOnlyPostFallback =
            DeepFissionAllowLegacyAuthorityOnly ||
            (PostPhaseReport.GuardedByAuthorityLane &&
             IsCarryPreloadRewriteResidual(
                 PostPhaseReport.ResidualRiskReason));
        if (AllowAuthorityOnlyPostFallback &&
            PostPhaseReport.GuardedByAuthorityLane) {
          errs() << "deepFission: [DF][PHASE] event=compat.legacy_authority_path func="
                 << StageFunc->getName() << " stage=" << StageIndex
                 << " reason="
                 << (PostPhaseReport.ResidualRiskReason.empty()
                         ? StringRef("risk_persists_after_lowering")
                         : StringRef(PostPhaseReport.ResidualRiskReason))
                 << "\n";
        } else {
          if (!PostPhaseReport.ResidualRiskReason.empty()) {
            errs() << "deepFission: [DF][PHASE] event=reject func="
                   << StageFunc->getName() << " stage=" << StageIndex
                   << " reason=" << PostPhaseReport.ResidualRiskReason << "\n";
          }
          errs() << "deepFission: [DF][PHASE] event=reject func="
                 << StageFunc->getName() << " stage=" << StageIndex
                 << " reason=risk_persists_after_lowering\n";
          return nullptr;
        }
      }
    } else {
      errs() << "deepFission: [DF][PHASE] event=postcheck.summary func="
             << StageFunc->getName() << " stage=" << StageIndex
             << " applied=0"
             << " risk=0"
             << " guarded=" << (PhaseReport.GuardedByAuthorityLane ? 1 : 0)
             << " early_authority_shape="
             << (PhaseReport.HasEarlyAuthorityReturnShape ? 1 : 0)
             << " subregion_rewritten="
             << (PhaseReport.HasMicroKernelRewrite ? 1 : 0)
             << " spill_fanout=" << (PhaseReport.HasSpillFanout ? 1 : 0)
             << " residual_reason="
             << (PhaseReport.ResidualRiskReason.empty()
                     ? StringRef("none")
                     : StringRef(PhaseReport.ResidualRiskReason))
             << " axis=" << authorityAxisKindName(PhaseReport.Axis)
             << " axis_modulo=" << PhaseReport.AxisModulo << "\n";
    }

    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] validating stabilized final stage " << StageFunc->getName()
           << " cloned from " << Info.WorkingStage->getName() << "\n";
    errs() << "deepFission: [DF][FINAL] event=final_stage.validate.start stage_func="
           << StageFunc->getName() << " stage=" << StageIndex
           << " working_stage="
           << (Info.WorkingStage ? Info.WorkingStage->getName()
                                 : StringRef("<null>"))
           << "\n";
    (void)canonicalizeSharedGlobalBasesForCBE(StageFunc);
    (void)simplifyDeadControlFlow(StageFunc);
    (void)pruneUnreachableBlocks(StageFunc);
    (void)repairNonDominatingAllocaLoads(StageFunc);
    if (!stabilizeBarrierFreeStageForCBE(StageFunc, true))
      return nullptr;
    if (!collectReferencedSharedGlobals(StageFunc).empty()) {
      errs() << "deepFission: final stage still references shared globals: "
             << StageFunc->getName() << "\n";
      return nullptr;
    }
    if (!collectReferencedSpillGlobals(StageFunc).empty()) {
      errs() << "deepFission: final stage still references spill globals: "
             << StageFunc->getName() << "\n";
      return nullptr;
    }
    if (!collectSyncMarkersInFunction(StageFunc).empty()) {
      errs() << "deepFission: final stage still contains reachable barriers: "
             << StageFunc->getName() << "\n";
      return nullptr;
    }
    if (!validateFinalStageSemantics(StageFunc))
      return nullptr;
    if (!verifyFunctionOrReport("createFinalStageFunction", StageFunc))
      return nullptr;
    errs() << "deepFission: [DF][FINAL] event=final_stage.validate.done stage_func="
           << StageFunc->getName() << " stage=" << StageIndex << "\n";
    return StageFunc;
  }

  static Value *getCallArgForNamedParam(CallInst *I, Function *Callee,
                                        StringRef Name) {
    if (!I || !Callee)
      return nullptr;
    unsigned MaxArgs = std::min<unsigned>(I->arg_size(), Callee->arg_size());
    auto ArgIt = Callee->arg_begin();
    for (unsigned AI = 0; AI < MaxArgs; ++AI, ++ArgIt) {
      if (ArgIt->getName() == Name)
        return I->getArgOperand(AI);
    }
    return nullptr;
  }

  static bool pointerValueHasLoadUse(Value *Root) {
    if (!Root)
      return false;
    SmallVector<Value *, 16> WorkList;
    SmallPtrSet<Value *, 32> Visited;
    WorkList.push_back(Root);
    while (!WorkList.empty()) {
      Value *V = WorkList.pop_back_val();
      if (!Visited.insert(V).second)
        continue;
      for (User *U : V->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I)
          continue;
        if (isa<LoadInst>(I))
          return true;
        if (auto *SI = dyn_cast<StoreInst>(I)) {
          if (SI->getPointerOperand() == V)
            continue;
        }
        if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I) ||
            isa<AddrSpaceCastInst>(I) || isa<SelectInst>(I) ||
            isa<PHINode>(I))
          WorkList.push_back(I);
      }
    }
    return false;
  }

  static bool stageReadsSpillArg(Function *StageFunc, unsigned SpillLocalIdx) {
    if (!StageFunc)
      return false;
    std::string ArgName = "spill." + std::to_string(SpillLocalIdx);
    Argument *SpillArg = findNamedArgument(StageFunc, ArgName);
    if (!SpillArg)
      return false;
    return pointerValueHasLoadUse(SpillArg);
  }

  Loop *findThreadLoopForKernelCall(CallInst *I, LoopInfo &LI) {
    if (!I || !I->getCalledFunction())
      return nullptr;
    Loop *ThreadLoop = LI.getLoopFor(I->getParent());
    if (!ThreadLoop)
      return nullptr;

    auto IsDynamicIndex = [](Value *V) -> bool {
      if (auto *CI = dyn_cast_or_null<ConstantInt>(V))
        return CI->getSExtValue() != 0;
      return V != nullptr;
    };

    unsigned ActiveThreadDims = 0;
    Function *Callee = I->getCalledFunction();
    Value *TidX = getCallArgForNamedParam(I, Callee, "threadIdx.x");
    Value *TidY = getCallArgForNamedParam(I, Callee, "threadIdx.y");
    Value *TidZ = getCallArgForNamedParam(I, Callee, "threadIdx.z");
    if (IsDynamicIndex(TidX))
      ActiveThreadDims++;
    if (IsDynamicIndex(TidY))
      ActiveThreadDims++;
    if (IsDynamicIndex(TidZ))
      ActiveThreadDims++;

    for (unsigned Depth = 1; Depth < ActiveThreadDims; ++Depth) {
      Loop *Parent = ThreadLoop->getParentLoop();
      if (!Parent)
        break;
      ThreadLoop = Parent;
    }
    return ThreadLoop;
  }

  Value *computeThreadsPerBlock(CallInst *I, Function *Seed, IRBuilder<> &B) {
    auto *I64Ty = Type::getInt64Ty(B.getContext());
    auto ToI64 = [&](Value *V) -> Value * {
      if (!V)
        return ConstantInt::get(I64Ty, 1);
      if (V->getType()->isIntegerTy(64))
        return V;
      if (!V->getType()->isIntegerTy())
        return ConstantInt::get(I64Ty, 1);
      return B.CreateZExtOrTrunc(V, I64Ty, "df.threads.ext");
    };

    Value *BDx = ToI64(getCallArgForNamedParam(I, Seed, "blockDim.x"));
    Value *BDy = ToI64(getCallArgForNamedParam(I, Seed, "blockDim.y"));
    Value *BDz = ToI64(getCallArgForNamedParam(I, Seed, "blockDim.z"));
    return B.CreateMul(BDx, B.CreateMul(BDy, BDz, "df.threads.yz"),
                       "df.threads.total");
  }

  Value *computeBlocksPerGrid(CallInst *I, Function *Seed, IRBuilder<> &B) {
    auto *I64Ty = Type::getInt64Ty(B.getContext());
    auto ToI64 = [&](Value *V) -> Value * {
      if (!V)
        return ConstantInt::get(I64Ty, 1);
      if (V->getType()->isIntegerTy(64))
        return V;
      if (!V->getType()->isIntegerTy())
        return ConstantInt::get(I64Ty, 1);
      return B.CreateZExtOrTrunc(V, I64Ty, "df.blocks.ext");
    };

    Value *GDx = ToI64(getCallArgForNamedParam(I, Seed, "gridDim.x"));
    Value *GDy = ToI64(getCallArgForNamedParam(I, Seed, "gridDim.y"));
    Value *GDz = ToI64(getCallArgForNamedParam(I, Seed, "gridDim.z"));
    return B.CreateMul(GDx, B.CreateMul(GDy, GDz, "df.blocks.yz"),
                       "df.blocks.total");
  }

  Value *computeLinearBlockIndex(CallInst *I, Function *Seed, IRBuilder<> &B) {
    auto *I64Ty = Type::getInt64Ty(B.getContext());
    auto ToI64 = [&](Value *V, uint64_t DefaultValue) -> Value * {
      if (!V)
        return ConstantInt::get(I64Ty, DefaultValue);
      if (V->getType()->isIntegerTy(64))
        return V;
      if (!V->getType()->isIntegerTy())
        return ConstantInt::get(I64Ty, DefaultValue);
      return B.CreateZExtOrTrunc(V, I64Ty, "df.blockidx.ext");
    };

    Value *BIdxX = ToI64(getCallArgForNamedParam(I, Seed, "blockIdx.x"), 0);
    Value *BIdxY = ToI64(getCallArgForNamedParam(I, Seed, "blockIdx.y"), 0);
    Value *BIdxZ = ToI64(getCallArgForNamedParam(I, Seed, "blockIdx.z"), 0);
    Value *GDx = ToI64(getCallArgForNamedParam(I, Seed, "gridDim.x"), 1);
    Value *GDy = ToI64(getCallArgForNamedParam(I, Seed, "gridDim.y"), 1);

    Value *YZ = B.CreateAdd(BIdxY, B.CreateMul(GDy, BIdxZ, "df.blockidx.yz.mul"),
                            "df.blockidx.yz");
    return B.CreateAdd(BIdxX,
                       B.CreateMul(GDx, YZ, "df.blockidx.linear.mul"),
                       "df.blockidx.linear");
  }

  Value *createHeapBuffer(IRBuilder<> &B, Module &M, Type *ElemTy,
                          Value *NumElements, const Twine &Name) {
    if (!ElemTy || !NumElements)
      return nullptr;

    const DataLayout &DL = M.getDataLayout();
    Type *IntPtrTy = DL.getIntPtrType(M.getContext());
    Value *Count = NumElements;
    if (Count->getType() != IntPtrTy)
      Count = B.CreateZExtOrTrunc(Count, IntPtrTy, Name + ".count");

    uint64_t ElemSizeValue = DL.getTypeAllocSize(ElemTy);
    Value *ElemSize = ConstantInt::get(IntPtrTy, ElemSizeValue);
    Value *ByteCount = B.CreateMul(Count, ElemSize, Name + ".bytes");

    FunctionCallee MallocFn = M.getOrInsertFunction(
        "malloc", FunctionType::get(Type::getInt8PtrTy(M.getContext()),
                                     {IntPtrTy}, false));
    CallInst *Raw =
        B.CreateCall(MallocFn, {ByteCount}, Name + ".raw");
    return B.CreateBitCast(Raw, ElemTy->getPointerTo(), Name);
  }

  void emitHeapBufferFree(IRBuilder<> &B, Module &M, Value *Buffer) {
    if (!Buffer)
      return;
    FunctionCallee FreeFn = M.getOrInsertFunction(
        "free", FunctionType::get(Type::getVoidTy(M.getContext()),
                                   {Type::getInt8PtrTy(M.getContext())},
                                   false));
    Value *Raw = Buffer;
    if (Raw->getType() != Type::getInt8PtrTy(M.getContext()))
      Raw = B.CreateBitCast(Raw, Type::getInt8PtrTy(M.getContext()),
                            "df.free.cast");
    B.CreateCall(FreeFn, {Raw});
  }

  void initializeActiveMask(BasicBlock *Entering, BasicBlock *FirstPreheader,
                            Value *ActiveMask, Value *ThreadsPerBlock) {
    if (!Entering || !FirstPreheader || !ActiveMask || !ThreadsPerBlock)
      return;

    Function *F = Entering->getParent();
    LLVMContext &Ctx = F->getContext();
    auto *I64Ty = Type::getInt64Ty(Ctx);
    BasicBlock *InitHdr =
        BasicBlock::Create(Ctx, "df.mask.init.hdr", F, FirstPreheader);
    BasicBlock *InitBody =
        BasicBlock::Create(Ctx, "df.mask.init.body", F, FirstPreheader);
    BasicBlock *InitDone =
        BasicBlock::Create(Ctx, "df.mask.init.done", F, FirstPreheader);

    Entering->getTerminator()->replaceUsesOfWith(FirstPreheader, InitHdr);

    IRBuilder<> HdrBuilder(InitHdr);
    PHINode *Idx = HdrBuilder.CreatePHI(I64Ty, 2, "df.mask.idx");
    Idx->addIncoming(ConstantInt::get(I64Ty, 0), Entering);
    Value *Limit = ThreadsPerBlock;
    if (!Limit->getType()->isIntegerTy(64))
      Limit = HdrBuilder.CreateZExtOrTrunc(Limit, I64Ty, "df.mask.limit");
    Value *Cond = HdrBuilder.CreateICmpULT(Idx, Limit, "df.mask.more");
    HdrBuilder.CreateCondBr(Cond, InitBody, InitDone);

    IRBuilder<> BodyBuilder(InitBody);
    Value *Slot =
        BodyBuilder.CreateInBoundsGEP(Type::getInt1Ty(Ctx), ActiveMask, Idx,
                                      "df.mask.slot");
    BodyBuilder.CreateStore(ConstantInt::getTrue(Type::getInt1Ty(Ctx)), Slot);
    Value *Next =
        BodyBuilder.CreateAdd(Idx, ConstantInt::get(I64Ty, 1), "df.mask.next");
    BodyBuilder.CreateBr(InitHdr);
    Idx->addIncoming(Next, InitBody);

    IRBuilder<> DoneBuilder(InitDone);
    DoneBuilder.CreateBr(FirstPreheader);
  }

  static bool isDriverExitSuccessor(BasicBlock *BB) {
    if (!BB)
      return false;
    if (BB->getName().startswith("df.exit") ||
        BB->getName().startswith("df.active.exit") ||
        BB->getName().startswith("df.boundary.exit"))
      return true;
    auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Br || Br->isConditional())
      return false;
    BasicBlock *Succ = Br->getSuccessor(0);
    return Succ && (Succ->getName().startswith("df.exit") ||
                    Succ->getName().startswith("df.active.exit") ||
                    Succ->getName().startswith("df.boundary.exit"));
  }

  static Value *materializeStage0MaskPredicate(
      IRBuilder<> &B, const Stage0EntrySemanticInfo *Stage0Entry,
      CallInst *Stage0Call, Value *LinearThreadIdx) {
    if (!Stage0Entry || !Stage0Entry->PredicateBranch || !Stage0Call ||
        !LinearThreadIdx) {
      errs() << "deepFission: df.mask predicate skip reason=missing_inputs\n";
      errs() << "deepFission: [DF][STAGE0] event=mask_predicate.skip reason=missing_inputs\n";
      return nullptr;
    }

    Function *Stage0Func = Stage0Entry->PredicateBranch->getFunction();
    if (!Stage0Func) {
      errs() << "deepFission: df.mask predicate skip reason=missing_stage0_func\n";
      errs() << "deepFission: [DF][STAGE0] event=mask_predicate.skip reason=missing_stage0_func\n";
      return nullptr;
    }
    errs() << "deepFission: [DF][STAGE0] event=mask_predicate.start stage_func="
           << Stage0Func->getName()
           << " stage=" << getStageIndexForLog(Stage0Func)
           << " caller="
           << (Stage0Call->getFunction() ? Stage0Call->getFunction()->getName()
                                         : StringRef("<null>"))
           << "\n";
    BlockIndexBiasInfo Stage0Bias;
    if (Stage0Entry)
      Stage0Bias = collectStage0BlockIndexBias(*Stage0Entry);

    auto *PredicateBranch = Stage0Entry->PredicateBranch;
    BasicBlock *PredicateBlock = PredicateBranch->getParent();
    if (!PredicateBranch->isConditional() || !PredicateBlock) {
      errs() << "deepFission: df.mask predicate skip func=" << Stage0Func->getName()
             << " reason=nonconditional_predicate\n";
      return nullptr;
    }

    BasicBlock *InsertBB = B.GetInsertBlock();
    auto InsertIt = B.GetInsertPoint();
    if (!InsertBB) {
      errs() << "deepFission: df.mask predicate skip func=" << Stage0Func->getName()
             << " reason=invalid_insert_point\n";
      return nullptr;
    }
    Instruction *InsertBefore =
        (InsertIt == InsertBB->end()) ? nullptr : &*InsertIt;

    Function *CallerF = Stage0Call->getFunction();
    if (!CallerF) {
      errs() << "deepFission: df.mask predicate skip func=" << Stage0Func->getName()
             << " reason=missing_caller\n";
      return nullptr;
    }
    DominatorTree CallerDT(*CallerF);

    auto ValueAvailableAtUse = [&](Value *V) -> bool {
      if (!V)
        return false;
      if (isa<Argument>(V) || isa<Constant>(V) || isa<GlobalValue>(V))
        return true;
      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getFunction() != CallerF)
        return false;
      if (I->getParent() == InsertBB) {
        if (!InsertBefore)
          return true;
        if (I == InsertBefore)
          return true;
        for (Instruction &Cur : *I->getParent()) {
          if (&Cur == I)
            return true;
          if (&Cur == InsertBefore)
            return false;
        }
        return false;
      }
      if (InsertBefore)
        return CallerDT.dominates(I, InsertBefore);
      return CallerDT.dominates(I->getParent(), InsertBB);
    };

    DenseMap<const Value *, Value *> CallerExprCache;
    std::function<Value *(Value *)> materializeCallerValueAtUse =
        [&](Value *V) -> Value * {
      if (!V)
        return nullptr;
      if (auto It = CallerExprCache.find(V); It != CallerExprCache.end())
        return It->second;
      if (ValueAvailableAtUse(V))
        return CallerExprCache[V] = V;
      if (isa<Constant>(V) || isa<Argument>(V) || isa<GlobalValue>(V))
        return CallerExprCache[V] = V;

      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getFunction() != CallerF)
        return nullptr;

      Value *NewV = nullptr;
      if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        Value *L = materializeCallerValueAtUse(BO->getOperand(0));
        Value *R = materializeCallerValueAtUse(BO->getOperand(1));
        if (!L || !R)
          return nullptr;
        NewV =
            B.CreateBinOp(BO->getOpcode(), L, R, I->getName() + ".df.driver");
      } else if (auto *CI = dyn_cast<CastInst>(I)) {
        Value *Src = materializeCallerValueAtUse(CI->getOperand(0));
        if (!Src)
          return nullptr;
        NewV = B.CreateCast(CI->getOpcode(), Src, CI->getType(),
                            I->getName() + ".df.driver");
      } else if (auto *Cmp = dyn_cast<ICmpInst>(I)) {
        Value *L = materializeCallerValueAtUse(Cmp->getOperand(0));
        Value *R = materializeCallerValueAtUse(Cmp->getOperand(1));
        if (!L || !R)
          return nullptr;
        NewV = B.CreateICmp(Cmp->getPredicate(), L, R,
                            I->getName() + ".df.driver");
      } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
        Value *C = materializeCallerValueAtUse(Sel->getCondition());
        Value *T = materializeCallerValueAtUse(Sel->getTrueValue());
        Value *F = materializeCallerValueAtUse(Sel->getFalseValue());
        if (!C || !T || !F)
          return nullptr;
        NewV = B.CreateSelect(C, T, F, I->getName() + ".df.driver");
      } else {
        errs() << "deepFission: df.mask caller remat skip func="
               << CallerF->getName() << " inst=" << *I
               << " reason=unsupported_caller_inst\n";
        return nullptr;
      }

      CallerExprCache[V] = NewV;
      return NewV;
    };

    DenseMap<const Value *, Value *> VMap;
    auto mapCallerArgByName = [&](StringRef Name) -> Value * {
      if (Name.startswith("blockIdx.")) {
        if (Value *Adapted =
                buildAdaptedBlockIndexArg(B, Stage0Call,
                                         Stage0Call->getCalledFunction(), Name,
                                         &Stage0Bias))
          return Adapted;
      }
      for (auto AI = Stage0Func->arg_begin(), AE = Stage0Func->arg_end();
           AI != AE; ++AI) {
        if (AI->getName() == Name) {
          if (AI->getArgNo() >= Stage0Call->arg_size())
            return (Value *)nullptr;
          Value *Mapped = Stage0Call->getArgOperand(AI->getArgNo());
          if (!ValueAvailableAtUse(Mapped))
            Mapped = materializeCallerValueAtUse(Mapped);
          return Mapped;
        }
      }
      return (Value *)nullptr;
    };

    Value *BDx = mapCallerArgByName("blockDim.x");
    Value *BDy = mapCallerArgByName("blockDim.y");
    Value *BDz = mapCallerArgByName("blockDim.z");
    if (!BDx || !BDy || !BDz) {
      errs() << "deepFission: df.mask predicate skip func=" << Stage0Func->getName()
             << " reason=missing_blockdim_arg\n";
      errs() << "deepFission: [DF][STAGE0] event=mask_predicate.skip func="
             << Stage0Func->getName() << " stage="
             << getStageIndexForLog(Stage0Func)
             << " reason=missing_blockdim_arg\n";
      return nullptr;
    }

    auto *I64Ty = Type::getInt64Ty(B.getContext());
    auto toI64 = [&](Value *V, const Twine &Name) -> Value * {
      if (V->getType()->isIntegerTy(64))
        return V;
      return B.CreateZExtOrTrunc(V, I64Ty, Name);
    };
    Value *BDx64 = toI64(BDx, "df.mask.bdx64");
    Value *BDy64 = toI64(BDy, "df.mask.bdy64");
    Value *ThreadX64 = B.CreateURem(LinearThreadIdx, BDx64, "df.mask.tidx.x64");
    Value *YZ64 = B.CreateUDiv(LinearThreadIdx, BDx64, "df.mask.tidx.yz64");
    Value *ThreadY64 = B.CreateURem(YZ64, BDy64, "df.mask.tidx.y64");
    Value *ThreadZ64 = B.CreateUDiv(YZ64, BDy64, "df.mask.tidx.z64");

    auto mapStage0Argument = [&](Argument &Arg) -> Value * {
      if (auto It = VMap.find(&Arg); It != VMap.end())
        return It->second;

      Value *Mapped = nullptr;
      if (Arg.getName() == "threadIdx.x")
        Mapped = B.CreateZExtOrTrunc(ThreadX64, Arg.getType(), "df.mask.tidx.x");
      else if (Arg.getName() == "threadIdx.y")
        Mapped = B.CreateZExtOrTrunc(ThreadY64, Arg.getType(), "df.mask.tidx.y");
      else if (Arg.getName() == "threadIdx.z")
        Mapped = B.CreateZExtOrTrunc(ThreadZ64, Arg.getType(), "df.mask.tidx.z");
      else if (!Arg.getName().empty()) {
        Mapped = mapCallerArgByName(Arg.getName());
        if (Mapped && Arg.getName().startswith("blockIdx.")) {
          errs() << "deepFission: df.mask arg remat func="
                 << Stage0Func->getName() << " arg=" << Arg.getName()
                 << " mapping=by_name";
          if (Stage0Bias.LowerThreadOffsetByBlockIdx.count(Arg.getName()))
            errs() << " adapted=1";
          else
            errs() << " adapted=0";
          errs() << "\n";
        }
      }
      if (!Mapped && Arg.getArgNo() < Stage0Call->arg_size()) {
        Mapped = Stage0Call->getArgOperand(Arg.getArgNo());
        if (!ValueAvailableAtUse(Mapped))
          Mapped = materializeCallerValueAtUse(Mapped);
        if (Arg.getName().startswith("blockIdx.")) {
          errs() << "deepFission: df.mask arg remat func="
                 << Stage0Func->getName() << " arg=" << Arg.getName()
                 << " mapping=by_position adapted=0\n";
        }
      }

      if (Mapped)
        VMap[&Arg] = Mapped;
      return Mapped;
    };

    DenseMap<const Value *, Value *> Cache;
    std::function<Value *(Value *)> cloneExpr = [&](Value *V) -> Value * {
      if (auto It = Cache.find(V); It != Cache.end())
        return It->second;
      if (auto It = VMap.find(V); It != VMap.end())
        return Cache[V] = It->second;
      if (auto *Arg = dyn_cast<Argument>(V)) {
        Value *Mapped = mapStage0Argument(*Arg);
        if (!Mapped) {
          errs() << "deepFission: df.mask predicate skip func="
                 << Stage0Func->getName() << " arg=" << Arg->getName()
                 << " reason=arg_remat_failed\n";
          return nullptr;
        }
        return Cache[V] = Mapped;
      }
      if (isa<Constant>(V))
        return Cache[V] = V;
      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getFunction() != Stage0Func)
        return nullptr;
      if (I->getParent() != PredicateBlock) {
        errs() << "deepFission: df.mask predicate skip func="
               << Stage0Func->getName() << " inst=" << *I
               << " reason=nonlocal_stage0_inst\n";
        return nullptr;
      }

      Value *NewV = nullptr;
      if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        Value *L = cloneExpr(BO->getOperand(0));
        Value *R = cloneExpr(BO->getOperand(1));
        if (!L || !R)
          return nullptr;
        NewV = B.CreateBinOp(BO->getOpcode(), L, R, I->getName() + ".df.mask");
      } else if (auto *CI = dyn_cast<CastInst>(I)) {
        Value *Src = cloneExpr(CI->getOperand(0));
        if (!Src)
          return nullptr;
        NewV = B.CreateCast(CI->getOpcode(), Src, CI->getType(),
                            I->getName() + ".df.mask");
      } else if (auto *Cmp = dyn_cast<ICmpInst>(I)) {
        Value *L = cloneExpr(Cmp->getOperand(0));
        Value *R = cloneExpr(Cmp->getOperand(1));
        if (!L || !R)
          return nullptr;
        NewV = B.CreateICmp(Cmp->getPredicate(), L, R, I->getName() + ".df.mask");
      } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
        Value *C = cloneExpr(Sel->getCondition());
        Value *T = cloneExpr(Sel->getTrueValue());
        Value *F = cloneExpr(Sel->getFalseValue());
        if (!C || !T || !F)
          return nullptr;
        NewV = B.CreateSelect(C, T, F, I->getName() + ".df.mask");
      } else {
        errs() << "deepFission: df.mask predicate skip func="
               << Stage0Func->getName() << " inst=" << *I
               << " reason=unsupported_stage0_inst\n";
        return nullptr;
      }
      Cache[V] = NewV;
      return NewV;
    };

    Value *PredCond = cloneExpr(PredicateBranch->getCondition());
    if (!PredCond) {
      errs() << "deepFission: df.mask predicate skip func=" << Stage0Func->getName()
             << " reason=predicate_clone_failed\n";
      return nullptr;
    }
    Value *Active =
        Stage0Entry->ActiveSuccessorIndex == 0
            ? PredCond
            : B.CreateNot(PredCond, "df.mask.active");
    errs() << "deepFission: df.mask predicate materialized stage_func="
           << Stage0Func->getName() << " caller=" << CallerF->getName()
           << " pred_bb=" << PredicateBlock->getName()
           << " active_succ=" << Stage0Entry->ActiveSuccessorIndex << "\n";
    errs() << "deepFission: [DF][STAGE0] event=mask_predicate.materialized stage_func="
           << Stage0Func->getName() << " stage="
           << getStageIndexForLog(Stage0Func) << " caller="
           << CallerF->getName() << " pred_bb=" << PredicateBlock->getName()
           << " active_succ=" << Stage0Entry->ActiveSuccessorIndex << "\n";
    return Active;
  }

  void initializeActiveMask(BasicBlock *Entering, BasicBlock *FirstPreheader,
                            Value *ActiveMask, Value *ThreadsPerBlock,
                            const std::optional<Stage0EntrySemanticInfo> &Stage0Entry,
                            CallInst *Stage0Call) {
    if (!Entering || !FirstPreheader || !ActiveMask || !ThreadsPerBlock)
      return;

    Function *F = Entering->getParent();
    LLVMContext &Ctx = F->getContext();
    auto *I64Ty = Type::getInt64Ty(Ctx);
    BasicBlock *InitHdr =
        BasicBlock::Create(Ctx, "df.mask.init.hdr", F, FirstPreheader);
    BasicBlock *InitBody =
        BasicBlock::Create(Ctx, "df.mask.init.body", F, FirstPreheader);
    BasicBlock *InitDone =
        BasicBlock::Create(Ctx, "df.mask.init.done", F, FirstPreheader);

    Entering->getTerminator()->replaceUsesOfWith(FirstPreheader, InitHdr);

    IRBuilder<> HdrBuilder(InitHdr);
    PHINode *Idx = HdrBuilder.CreatePHI(I64Ty, 2, "df.mask.idx");
    Idx->addIncoming(ConstantInt::get(I64Ty, 0), Entering);
    Value *Limit = ThreadsPerBlock;
    if (!Limit->getType()->isIntegerTy(64))
      Limit = HdrBuilder.CreateZExtOrTrunc(Limit, I64Ty, "df.mask.limit");
    Value *Cond = HdrBuilder.CreateICmpULT(Idx, Limit, "df.mask.more");
    HdrBuilder.CreateCondBr(Cond, InitBody, InitDone);

    IRBuilder<> BodyBuilder(InitBody);
    Value *Slot =
        BodyBuilder.CreateInBoundsGEP(Type::getInt1Ty(Ctx), ActiveMask, Idx,
                                      "df.mask.slot");
    Value *ActiveValue = nullptr;
    if (Stage0Entry && Stage0Call)
      ActiveValue = materializeStage0MaskPredicate(BodyBuilder, &*Stage0Entry,
                                                   Stage0Call, Idx);
    if (!ActiveValue) {
      errs() << "deepFission: df.mask init fallback func=" << F->getName()
             << " stage0_entry=" << (Stage0Entry ? "yes" : "no")
             << " stage0_call="
             << (Stage0Call && Stage0Call->getCalledFunction()
                     ? Stage0Call->getCalledFunction()->getName()
                     : StringRef("<null>"))
             << "\n";
      errs() << "deepFission: [DF][STAGE0] event=mask_init.fallback func="
             << F->getName() << " stage=" << getStageIndexForLog(F)
             << " stage0_entry=" << (Stage0Entry ? 1 : 0)
             << " stage0_call="
             << (Stage0Call && Stage0Call->getCalledFunction()
                     ? Stage0Call->getCalledFunction()->getName()
                     : StringRef("<null>"))
             << "\n";
      ActiveValue = ConstantInt::getTrue(Type::getInt1Ty(Ctx));
    } else {
      errs() << "deepFission: df.mask init using predicate func=" << F->getName()
             << "\n";
      errs() << "deepFission: [DF][STAGE0] event=mask_init.predicate func="
             << F->getName() << " stage=" << getStageIndexForLog(F) << "\n";
    }
    BodyBuilder.CreateStore(ActiveValue, Slot);
    Value *Next =
        BodyBuilder.CreateAdd(Idx, ConstantInt::get(I64Ty, 1), "df.mask.next");
    BodyBuilder.CreateBr(InitHdr);
    Idx->addIncoming(Next, InitBody);

    IRBuilder<> DoneBuilder(InitDone);
    DoneBuilder.CreateBr(FirstPreheader);
  }

  CallInst *replaceCallWithStageCall(
      CallInst *OldCall, Function *StageFunc,
      DenseMap<GlobalVariable *, Value *> &SpillSlots,
      DenseMap<GlobalVariable *, Value *> &SharedBases,
      ArrayRef<GlobalVariable *> UsedSpills,
      ArrayRef<GlobalVariable *> UsedSharedGlobals, Value *ActiveMask,
      const BlockIndexBiasInfo *Stage0Bias = nullptr) {
    if (!OldCall || !StageFunc || !ActiveMask)
      return nullptr;

    IRBuilder<> B(OldCall);
    SmallVector<Value *, 16> Args;
    Function *OldCallee = OldCall->getCalledFunction();
    unsigned SeedArgCount = std::min<unsigned>(OldCall->arg_size(),
                                               StageFunc->arg_size());
    unsigned ExpectedArgCount = SeedArgCount + UsedSpills.size() +
                                UsedSharedGlobals.size() + 1;
    if (StageFunc->arg_size() < ExpectedArgCount) {
      errs() << "deepFission: [DF][CALLMAP] event=error func="
             << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                        : StringRef("<null>"))
             << " stage_func=" << StageFunc->getName()
             << " stage=" << getStageIndexForLog(StageFunc)
             << " reason=stage_arg_count_too_small expected="
             << ExpectedArgCount << " actual=" << StageFunc->arg_size()
             << "\n";
      return nullptr;
    }
    std::string StageIndexForLog = getStageIndexForLog(StageFunc);
    const bool AllowStageBlockIdxAdapt = Stage0Bias != nullptr;

    errs() << "deepFission: [DF][CALLMAP] event=call.start caller="
           << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                      : StringRef("<null>"))
           << " callee="
           << (OldCallee ? OldCallee->getName() : StringRef("<indirect>"))
           << " stage_func=" << StageFunc->getName()
           << " stage=" << StageIndexForLog
           << " seed_args=" << SeedArgCount
           << " spill_args=" << UsedSpills.size()
           << " shared_args=" << UsedSharedGlobals.size() << "\n";
    auto StageArgIt = StageFunc->arg_begin();
    for (unsigned AI = 0; AI < SeedArgCount; ++AI, ++StageArgIt) {
      Value *Arg = OldCall->getArgOperand(AI);
      bool AdaptedBlockIdx = false;
      if (StageArgIt->getName().startswith("blockIdx.")) {
        const bool HasBiasForDim =
            AllowStageBlockIdxAdapt &&
            Stage0Bias->LowerThreadOffsetByBlockIdx.count(
                StageArgIt->getName());
        if (AllowStageBlockIdxAdapt) {
          if (Value *Adapted = buildAdaptedBlockIndexArg(
                  B, OldCall, OldCallee, StageArgIt->getName(), Stage0Bias)) {
            Arg = Adapted;
            AdaptedBlockIdx = true;
          }
        }
        if (HasBiasForDim && !AdaptedBlockIdx) {
          errs() << "deepFission: [DF][STAGE0] event=driver_call.blockidx_scope.error caller="
                 << (OldCall->getFunction()
                         ? OldCall->getFunction()->getName()
                         : StringRef("<null>"))
                 << " stage_func=" << StageFunc->getName()
                 << " stage=" << StageIndexForLog
                 << " arg=" << StageArgIt->getName()
                 << " reason=required_blockidx_bias_not_applied\n";
          return nullptr;
        }
        errs() << "deepFission: [DF][STAGE0] event=driver_call.blockidx_scope caller="
               << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                          : StringRef("<null>"))
               << " stage_func=" << StageFunc->getName()
               << " stage=" << StageIndexForLog
               << " arg=" << StageArgIt->getName()
               << " adapted=" << (AdaptedBlockIdx ? 1 : 0)
               << " has_bias=" << (HasBiasForDim ? 1 : 0)
               << "\n";
      }
      errs() << "deepFission: [DF][CALLMAP] event=call.arg caller="
             << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                        : StringRef("<null>"))
             << " stage_func=" << StageFunc->getName()
             << " stage=" << StageIndexForLog
             << " arg_index=" << AI << " role=abi name="
             << StageArgIt->getName() << " adapted="
             << (AdaptedBlockIdx ? 1 : 0) << "\n";
      Args.push_back(Arg);
    }
    for (unsigned SpillLocalIdx = 0; SpillLocalIdx < UsedSpills.size();
         ++SpillLocalIdx) {
      GlobalVariable *GV = UsedSpills[SpillLocalIdx];
      auto It = SpillSlots.find(GV);
      if (It == SpillSlots.end())
        return nullptr;
      Args.push_back(It->second);
      errs() << "deepFission: [DF][CALLMAP] event=call.arg caller="
             << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                        : StringRef("<null>"))
             << " stage_func=" << StageFunc->getName()
             << " stage=" << StageIndexForLog
             << " arg_index=" << (SeedArgCount + SpillLocalIdx)
             << " role=spill spill_local_index=" << SpillLocalIdx
             << " spill_global=" << GV->getName() << "\n";
    }
    for (unsigned SharedLocalIdx = 0; SharedLocalIdx < UsedSharedGlobals.size();
         ++SharedLocalIdx) {
      GlobalVariable *GV = UsedSharedGlobals[SharedLocalIdx];
      auto It = SharedBases.find(GV);
      if (It == SharedBases.end())
        return nullptr;
      Args.push_back(It->second);
      errs() << "deepFission: [DF][CALLMAP] event=call.arg caller="
             << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                        : StringRef("<null>"))
             << " stage_func=" << StageFunc->getName()
             << " stage=" << StageIndexForLog
             << " arg_index="
             << (SeedArgCount + UsedSpills.size() + SharedLocalIdx)
             << " role=shared shared_global=" << GV->getName() << "\n";
    }
    Args.push_back(ActiveMask);
    errs() << "deepFission: [DF][CALLMAP] event=call.arg caller="
           << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                      : StringRef("<null>"))
           << " stage_func=" << StageFunc->getName()
           << " stage=" << StageIndexForLog << " arg_index="
           << (SeedArgCount + UsedSpills.size() + UsedSharedGlobals.size())
           << " role=active_mask\n";

    CallInst *NewCall = B.CreateCall(StageFunc, Args);
    NewCall->setCallingConv(OldCall->getCallingConv());
    errs() << "deepFission: [DF][CALLMAP] event=call.done caller="
           << (OldCall->getFunction() ? OldCall->getFunction()->getName()
                                      : StringRef("<null>"))
           << " stage_func=" << StageFunc->getName()
           << " stage=" << StageIndexForLog
           << " total_args=" << Args.size() << "\n";
    if (!OldCall->use_empty())
      OldCall->replaceAllUsesWith(NewCall);
    OldCall->eraseFromParent();
    return NewCall;
  }

  bool rewriteCallerForStageSequence(
      CallInst *SeedCall, Function *Seed,
      const std::vector<FinalStageInfo> &FinalStages) {
    if (!SeedCall || !Seed || FinalStages.empty() || !SeedCall->getParent())
      return false;
    errs() << "deepFission: [DF][CALLMAP] event=rewrite.start caller="
           << (SeedCall->getFunction() ? SeedCall->getFunction()->getName()
                                       : StringRef("<null>"))
           << " seed=" << Seed->getName() << " stages=" << FinalStages.size()
           << "\n";

    Function *Caller = SeedCall->getFunction();
    auto &LI = getAnalysis<LoopInfoWrapperPass>(*Caller).getLoopInfo();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>(*Caller).getDomTree();
    Loop *ThreadLoop = findThreadLoopForKernelCall(SeedCall, LI);
    if (!ThreadLoop) {
      errs() << "deepFission: skipping call not enclosed by thread loop: "
             << *SeedCall << "\n";
      return false;
    }

    if (!ThreadLoop->getLoopPreheader())
      (void)makeLoopPreheader(LI, DT, ThreadLoop);
    if (!ThreadLoop->getLoopPreheader()) {
      errs() << "deepFission: failed to materialize loop preheader for "
             << *SeedCall << "\n";
      return false;
    }

    Loop *AllocationLoop = ThreadLoop;
    while (Loop *Parent = AllocationLoop->getParentLoop())
      AllocationLoop = Parent;
    if (!AllocationLoop->getLoopPreheader())
      (void)makeLoopPreheader(LI, DT, AllocationLoop);
    if (!AllocationLoop->getLoopPreheader()) {
      errs() << "deepFission: failed to materialize allocation preheader for "
             << *SeedCall << "\n";
      return false;
    }
    BasicBlock *AllocationPreheader = AllocationLoop->getLoopPreheader();

    BasicBlock *Entering =
        ThreadLoop->getLoopPreheader()->getSinglePredecessor();
    if (!Entering) {
      errs() << "deepFission: loop without single preheader predecessor: "
             << *SeedCall << "\n";
      return false;
    }

    BasicBlock *AllocationExit = AllocationLoop->getExitBlock();
    if (!AllocationExit) {
      errs() << "deepFission: allocation loop without unique exit block: "
             << *SeedCall << "\n";
      return false;
    }

    std::vector<CallInst *> StageCalls(FinalStages.size(), nullptr);
    StageCalls.back() = SeedCall;
    Loop *PrevLoop = ThreadLoop;
    for (int StageIndex = static_cast<int>(FinalStages.size()) - 2; StageIndex >= 0;
         --StageIndex) {
      ValueToValueMapTy VMap;
      SmallVector<BasicBlock *, 4> NewBlocks;
      Loop *NewLoop = cloneLoopWithPreheader(
          PrevLoop->getLoopPreheader(), &Caller->getEntryBlock(), ThreadLoop,
          VMap, ".df.clone" + std::to_string(StageIndex), &LI, &DT, NewBlocks);
      remapInstructionsInBlocks(NewBlocks, VMap);
      Entering->getTerminator()->replaceUsesOfWith(PrevLoop->getLoopPreheader(),
                                                   NewLoop->getLoopPreheader());
      NewLoop->getHeader()->getTerminator()->replaceUsesOfWith(
          ThreadLoop->getExitBlock(), PrevLoop->getLoopPreheader());

      auto MapIt = VMap.find(SeedCall);
      auto *MappedCall =
          (MapIt == VMap.end()) ? nullptr : dyn_cast<CallInst>(MapIt->second);
      if (!MappedCall)
        return false;
      StageCalls[StageIndex] = MappedCall;
      PrevLoop = NewLoop;
    }

    IRBuilder<> AllocBuilder(AllocationPreheader->getTerminator());
    Value *ThreadsPerBlock =
        computeThreadsPerBlock(SeedCall, Seed, AllocBuilder);
    Value *BlocksPerGrid = computeBlocksPerGrid(SeedCall, Seed, AllocBuilder);
    Value *TotalStateSlots =
        AllocBuilder.CreateMul(ThreadsPerBlock, BlocksPerGrid,
                               "df.state.total.slots");
    auto *ActiveMask = createHeapBuffer(AllocBuilder, *Caller->getParent(),
                                        Type::getInt1Ty(Caller->getContext()),
                                        TotalStateSlots, "df.active_mask");
    if (!ActiveMask)
      return false;

    SmallVector<GlobalVariable *, 16> AllSpills;
    SmallPtrSet<GlobalVariable *, 16> SeenSpills;
    for (const FinalStageInfo &Info : FinalStages) {
      for (GlobalVariable *GV : Info.UsedSpills) {
        if (SeenSpills.insert(GV).second)
          AllSpills.push_back(GV);
      }
    }
    llvm::sort(AllSpills, [](GlobalVariable *L, GlobalVariable *R) {
      return L->getName() < R->getName();
    });

    DenseMap<GlobalVariable *, Value *> SpillSlots;
    for (GlobalVariable *GV : AllSpills) {
      auto *ArrTy = dyn_cast<ArrayType>(GV->getValueType());
      if (!ArrTy)
        return false;
      SpillSlots[GV] = createHeapBuffer(AllocBuilder, *Caller->getParent(),
                                        ArrTy->getElementType(),
                                        TotalStateSlots,
                                        GV->getName() + ".buf");
      if (!SpillSlots[GV])
        return false;
    }

    SmallVector<GlobalVariable *, 8> AllSharedGlobals;
    SmallPtrSet<GlobalVariable *, 8> SeenSharedGlobals;
    for (const FinalStageInfo &Info : FinalStages) {
      for (GlobalVariable *GV : Info.UsedSharedGlobals) {
        if (SeenSharedGlobals.insert(GV).second)
          AllSharedGlobals.push_back(GV);
      }
    }
    llvm::sort(AllSharedGlobals, [](GlobalVariable *L, GlobalVariable *R) {
      return L->getName() < R->getName();
    });

    DenseMap<GlobalVariable *, Value *> SharedBases;
    IRBuilder<NoFolder> SharedBuilder(Caller->getContext());
    SharedBuilder.SetInsertPoint(AllocationPreheader->getTerminator());
    for (GlobalVariable *GV : AllSharedGlobals) {
      auto *ArrTy = dyn_cast<ArrayType>(GV->getValueType());
      if (!ArrTy)
        return false;
      Type *ElemTy = ArrTy->getElementType();
      auto *ZeroArrTy = ArrayType::get(ElemTy, 0);
      Value *Cast = SharedBuilder.CreateBitCast(
          GV, ZeroArrTy->getPointerTo(GV->getAddressSpace()),
          GV->getName() + ".df.shared.arr");
      SharedBases[GV] = SharedBuilder.CreateInBoundsGEP(
          ZeroArrTy, Cast,
          {SharedBuilder.getInt64(0), SharedBuilder.getInt64(0)},
          GV->getName() + ".sharedMem.gep");
    }

    IRBuilder<> SliceBuilder(Entering->getTerminator());
    Value *BlockLinearIndex =
        computeLinearBlockIndex(SeedCall, Seed, SliceBuilder);
    Value *BlockStateBase =
        SliceBuilder.CreateMul(BlockLinearIndex, ThreadsPerBlock,
                               "df.state.block.base");
    Value *ActiveMaskSlice = SliceBuilder.CreateInBoundsGEP(
        Type::getInt1Ty(Caller->getContext()), ActiveMask, BlockStateBase,
        "df.active_mask.block");

    DenseMap<GlobalVariable *, Value *> SpillSlices;
    for (GlobalVariable *GV : AllSpills) {
      auto It = SpillSlots.find(GV);
      if (It == SpillSlots.end())
        return false;
      auto *ArrTy = dyn_cast<ArrayType>(GV->getValueType());
      if (!ArrTy)
        return false;
      SpillSlices[GV] = SliceBuilder.CreateInBoundsGEP(
          ArrTy->getElementType(), It->second, BlockStateBase,
          GV->getName() + ".block");
    }

    initializeActiveMask(Entering, PrevLoop->getLoopPreheader(), ActiveMaskSlice,
                         ThreadsPerBlock,
                         FinalStages.front().Stage0EntrySemantics,
                         SeedCall);

    BlockIndexBiasInfo Stage0Bias =
        collectStage0BlockIndexBias(FinalStages.front().Stage0EntrySemantics);
    for (const auto &BiasKV : Stage0Bias.LowerThreadOffsetByBlockIdx) {
      bool FoundSeedArg = false;
      for (Argument &SeedArg : Seed->args()) {
        if (SeedArg.getName() == BiasKV.getKey()) {
          FoundSeedArg = true;
          break;
        }
      }
      errs() << "deepFission: [DF][STAGE0] event=bias.scope_check seed="
             << Seed->getName() << " dim=" << BiasKV.getKey()
             << " lower_thread_offset=" << BiasKV.getValue()
             << " found_in_seed_abi=" << (FoundSeedArg ? 1 : 0) << "\n";
      if (!FoundSeedArg) {
        errs() << "deepFission: [DF][STAGE0] event=bias.scope_check.error seed="
               << Seed->getName() << " dim=" << BiasKV.getKey()
               << " reason=missing_seed_abi_operand\n";
        return false;
      }
    }

    errs() << "deepFission: [DF][SPILLFLOW] event=stages.summary seed="
           << Seed->getName() << " stages=" << FinalStages.size() << "\n";
    for (unsigned StageIndex = 0; StageIndex < FinalStages.size(); ++StageIndex) {
      const FinalStageInfo &Info = FinalStages[StageIndex];
      for (unsigned SpillLocalIdx = 0; SpillLocalIdx < Info.UsedSpills.size();
           ++SpillLocalIdx) {
        GlobalVariable *GV = Info.UsedSpills[SpillLocalIdx];
        bool Reads = stageReadsSpillArg(Info.StageFunc, SpillLocalIdx);
        errs() << "deepFission: [DF][SPILLFLOW] event=stage.spill seed="
               << Seed->getName() << " stage=" << StageIndex
               << " stage_func="
               << (Info.StageFunc ? Info.StageFunc->getName() : StringRef("<null>"))
               << " spill_local_index=" << SpillLocalIdx
               << " spill_global=" << GV->getName()
               << " consumer_reads=" << (Reads ? 1 : 0) << "\n";
      }
    }
    for (unsigned StageIndex = 1; StageIndex < FinalStages.size(); ++StageIndex) {
      const FinalStageInfo &Producer = FinalStages[StageIndex - 1];
      const FinalStageInfo &Consumer = FinalStages[StageIndex];
      SmallPtrSet<GlobalVariable *, 16> ProducerSpills;
      for (GlobalVariable *GV : Producer.UsedSpills)
        ProducerSpills.insert(GV);

      for (unsigned SpillLocalIdx = 0; SpillLocalIdx < Consumer.UsedSpills.size();
           ++SpillLocalIdx) {
        GlobalVariable *GV = Consumer.UsedSpills[SpillLocalIdx];
        bool ProducerHas = ProducerSpills.count(GV);
        bool ConsumerReads = stageReadsSpillArg(Consumer.StageFunc, SpillLocalIdx);
        bool ProducedEarlier = false;
        for (unsigned Prior = 0; Prior + 1 < StageIndex; ++Prior) {
          if (llvm::is_contained(FinalStages[Prior].UsedSpills, GV)) {
            ProducedEarlier = true;
            break;
          }
        }
        errs() << "deepFission: [DF][SPILLFLOW] event=edge seed="
               << Seed->getName() << " stage_from=" << (StageIndex - 1)
               << " stage_to=" << StageIndex << " spill_global="
               << GV->getName() << " producer_has=" << (ProducerHas ? 1 : 0)
               << " consumer_reads=" << (ConsumerReads ? 1 : 0)
               << " produced_earlier=" << (ProducedEarlier ? 1 : 0) << "\n";
        if (ConsumerReads && !ProducerHas && ProducedEarlier) {
          errs() << "deepFission: [DF][SPILLFLOW] event=edge.warn seed="
                 << Seed->getName() << " stage_from=" << (StageIndex - 1)
                 << " stage_to=" << StageIndex << " spill_global="
                 << GV->getName()
                 << " reason=consumer_reads_non_immediate_producer\n";
        }
        if (ConsumerReads && !ProducerHas && !ProducedEarlier) {
          errs() << "deepFission: [DF][SPILLFLOW] event=edge.error seed="
                 << Seed->getName() << " stage_from=" << (StageIndex - 1)
                 << " stage_to=" << StageIndex << " spill_global="
                 << GV->getName()
                 << " reason=consumer_reads_spill_without_prior_producer\n";
          return false;
        }
      }
    }

    for (unsigned StageIndex = 0; StageIndex < FinalStages.size(); ++StageIndex) {
      if (!replaceCallWithStageCall(StageCalls[StageIndex],
                                    FinalStages[StageIndex].StageFunc, SpillSlices,
                                    SharedBases,
                                    FinalStages[StageIndex].UsedSpills,
                                    FinalStages[StageIndex].UsedSharedGlobals,
                                    ActiveMaskSlice, &Stage0Bias))
        return false;
    }

    SmallVector<BasicBlock *, 8> CleanupPreds;
    for (BasicBlock *Pred : predecessors(AllocationExit)) {
      if (AllocationLoop->contains(Pred))
        CleanupPreds.push_back(Pred);
    }
    if (CleanupPreds.empty()) {
      errs() << "deepFission: allocation loop exit has no in-loop predecessors: "
             << *SeedCall << "\n";
      return false;
    }

    BasicBlock *CleanupBB = BasicBlock::Create(
        Caller->getContext(), "df.state.cleanup", Caller, AllocationExit);
    for (BasicBlock *Pred : CleanupPreds)
      Pred->getTerminator()->replaceUsesOfWith(AllocationExit, CleanupBB);

    IRBuilder<> FreeBuilder(CleanupBB);
    for (GlobalVariable *GV : AllSpills) {
      auto It = SpillSlots.find(GV);
      if (It == SpillSlots.end())
        return false;
      emitHeapBufferFree(FreeBuilder, *Caller->getParent(), It->second);
    }
    emitHeapBufferFree(FreeBuilder, *Caller->getParent(), ActiveMask);
    FreeBuilder.CreateBr(AllocationExit);

    errs() << "deepFission: [DF][CALLMAP] event=rewrite.done caller="
           << Caller->getName() << " seed=" << Seed->getName()
           << " stages=" << FinalStages.size() << "\n";
    return verifyFunctionOrReport("rewriteCallerForStageSequence", Caller);
  }

  void discoverCandidates(Module &M, std::vector<Function *> &SplitSeeds) {
    std::set<Function *> CandidateSet;
    for (Function &F : M) {
      if (!hasSerializedKernelABI(&F))
        continue;
      std::vector<Instruction *> Markers = collectSyncMarkersInFunction(&F);
      if (Markers.empty())
        continue;
      SplitSeeds.push_back(&F);
      CandidateSet.insert(&F);
    }

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      for (Instruction &I : instructions(F)) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee || !CandidateSet.count(Callee))
          continue;
        kernelCalls[Callee].insert(CI);
      }
    }

    SplitSeeds.erase(
        std::remove_if(SplitSeeds.begin(), SplitSeeds.end(),
                       [&](Function *F) {
                         auto It = kernelCalls.find(F);
                         return It == kernelCalls.end() || It->second.empty();
                       }),
        SplitSeeds.end());
  }

  void rollbackSeedArtifacts(Module &M, Function *Seed,
                             std::list<Function *> &WorkingStages,
                             std::vector<FinalStageInfo> &FinalStages) {
    if (!Seed)
      return;
    errs() << "deepFission: rolling back temporary artifacts for seed "
           << Seed->getName() << "\n";

    SmallPtrSet<Function *, 32> FunctionsToErase;
    for (FinalStageInfo &Info : FinalStages) {
      if (Info.StageFunc && Info.StageFunc->getParent())
        FunctionsToErase.insert(Info.StageFunc);
    }
    for (Function *WorkingStage : WorkingStages) {
      if (WorkingStage && WorkingStage->getParent())
        FunctionsToErase.insert(WorkingStage);
    }

    SmallVector<Function *, 32> OrderedFunctions(FunctionsToErase.begin(),
                                                 FunctionsToErase.end());
    for (Function *F : OrderedFunctions) {
      F->setSubprogram(nullptr);
      F->clearMetadata();
      F->dropAllReferences();
    }
    for (Function *F : OrderedFunctions) {
      if (!F->use_empty()) {
        errs() << "deepFission: rollback could not erase function still in use "
               << F->getName() << "\n";
        continue;
      }
      F->eraseFromParent();
    }

    std::string SpillPrefix = getDeepFissionBaseName(Seed) + ".df.spill.";
    SmallVector<GlobalVariable *, 16> SpillGlobals;
    for (GlobalVariable &GV : M.globals()) {
      if (GV.getName().startswith(SpillPrefix))
        SpillGlobals.push_back(&GV);
    }
    for (GlobalVariable *GV : SpillGlobals) {
      if (GV && GV->use_empty()) {
        errs() << "deepFission: deleting temporary spill global "
               << GV->getName() << "\n";
        GV->eraseFromParent();
      }
    }
  }

  bool runOnModule(Module &M) override {
    errs() << "deepFission: [" << DeepFissionBuildTag
           << "] runOnModule start module=" << M.getName() << "\n";
    std::vector<Function *> SplitSeeds;
    discoverCandidates(M, SplitSeeds);
    if (SplitSeeds.empty()) {
      errs() << "deepFission: no serialized kernels with reachable barriers\n";
      return false;
    }

    bool Changed = false;
    for (Function *Seed : SplitSeeds) {
      std::string SeedName = Seed->getName().str();
      auto KCIt = kernelCalls.find(Seed);
      if (KCIt == kernelCalls.end() || KCIt->second.empty())
        continue;

      std::vector<CallInst *> OriginalCallsites(KCIt->second.begin(),
                                                KCIt->second.end());
      Function *WorkingRoot = cloneSeedToWorkingStage(Seed);
      if (!WorkingRoot)
        continue;

      std::list<Function *> WorkingStages;
      WorkingStages.push_back(WorkingRoot);
      std::deque<std::list<Function *>::iterator> WorkList;
      WorkList.push_back(WorkingStages.begin());

      unsigned WorkingStageIndex = 1;
      unsigned NextSpillIndex = 0;
      unsigned SafetyCounter = 0;
      bool SeedChanged = false;
      bool SeedFailed = false;
      std::vector<FinalStageInfo> FinalStages;
      while (!WorkList.empty()) {
        auto StageIt = WorkList.front();
        WorkList.pop_front();
        Function *Current = *StageIt;
        if (!Current || Current->isDeclaration())
          continue;

        normalizeKernelForSplit(Current);
        if (!verifyFunctionOrReport("normalizeKernelForSplit", Current)) {
          SeedFailed = true;
          break;
        }

        std::vector<Instruction *> Markers =
            collectSyncMarkersInFunction(Current);
        errs() << "deepFission: work stage function=" << Current->getName()
               << " reachable markers=" << Markers.size() << "\n";
        errs() << "deepFission: [DF][SPLIT] event=workstage.markers func="
               << Current->getName() << " stage="
               << getStageIndexForLog(Current)
               << " reachable_markers=" << Markers.size() << "\n";
        if (Markers.empty())
          continue;

        auto &CurrentLI = getAnalysis<LoopInfoWrapperPass>(*Current).getLoopInfo();
        dumpMarkerList("work-stage candidate barriers", Markers, &CurrentLI);
        Instruction *ChosenMarker = nullptr;
        for (Instruction *Marker : Markers) {
          if (!Marker || !Marker->getParent())
            continue;
          ChosenMarker = Marker;
          break;
        }
        if (!ChosenMarker) {
          SeedFailed = true;
          break;
        }
        if (CurrentLI.getLoopFor(ChosenMarker->getParent())) {
          errs() << "deepFission: work stage function=" << Current->getName()
                 << " selected earliest loop-contained barrier\n";
        }

        Function *Next =
            splitFunction(M.getContext(), Current, ChosenMarker, Markers.size(),
                          WorkingStageIndex++, NextSpillIndex);
        if (!Next) {
          SeedFailed = true;
          break;
        }
        errs() << "deepFission: [DF][SPLIT] event=workstage.split func="
               << Current->getName() << " next=" << Next->getName()
               << " stage=" << getStageIndexForLog(Current)
               << " next_stage=" << getStageIndexForLog(Next) << "\n";

        auto NewIt = WorkingStages.insert(std::next(StageIt), Next);
        WorkList.push_back(StageIt);
        WorkList.push_back(NewIt);
        Changed = true;
        SeedChanged = true;

        if (++SafetyCounter > 256) {
          errs() << "deepFission: aborting due to excessive stage splits for "
                 << SeedName << "\n";
          SeedFailed = true;
          break;
        }
      }

      if (SeedFailed) {
        rollbackSeedArtifacts(M, Seed, WorkingStages, FinalStages);
        continue;
      }

      for (Function *WorkingStage : WorkingStages) {
        if (!stabilizeBarrierFreeWorkStage(WorkingStage)) {
          SeedFailed = true;
          break;
        }
        std::vector<Instruction *> ResidualMarkers =
            collectSyncMarkersInFunction(WorkingStage);
        errs() << "deepFission: finalized work stage "
               << WorkingStage->getName() << " reachable markers="
               << ResidualMarkers.size() << "\n";
        if (!ResidualMarkers.empty()) {
          errs() << "deepFission: refusing to finalize stage with reachable "
                    "barriers: "
                 << WorkingStage->getName() << "\n";
          SeedFailed = true;
          break;
        }

        FinalStageInfo Info;
        Info.WorkingStage = WorkingStage;
        Info.UsedSpills = collectReferencedSpillGlobals(WorkingStage);
        Info.UsedSharedGlobals = collectReferencedSharedGlobals(WorkingStage);
        bool AllowStageEntryLaneTermination = FinalStages.empty();
        Info.ExitSemantics = collectExplicitExitSemantics(
            WorkingStage, AllowStageEntryLaneTermination);
        errs() << "deepFission: [DF][FINAL] event=workstage.exit_semantics func="
               << WorkingStage->getName() << " stage="
               << getStageIndexForLog(WorkingStage) << " count="
               << Info.ExitSemantics.size() << "\n";
        for (const ExitSemanticInfo &ExitInfo : Info.ExitSemantics) {
          if (ExitInfo.Kind != ExitSemanticKind::PreserveActive)
            continue;
          if (!ExitInfo.RootReturn || !ExitInfo.RootReturn->getParent() ||
              ExitInfo.RootReturn->getParent()->getParent() != WorkingStage) {
            errs() << "deepFission: [DF][FINAL] event=workstage.exit_semantics.error func="
                   << WorkingStage->getName() << " stage="
                   << getStageIndexForLog(WorkingStage)
                   << " reason=preserve_active_root_missing_or_invalid\n";
            SeedFailed = true;
            break;
          }
          errs() << "deepFission: [DF][FINAL] event=workstage.preserve_root func="
                 << WorkingStage->getName() << " stage="
                 << getStageIndexForLog(WorkingStage) << " root="
                 << ExitInfo.RootReturn->getParent()->getName()
                 << " entry_edges=" << ExitInfo.EntryEdges.size() << "\n";
        }
        if (SeedFailed)
          break;
        // Only the first split stage may establish global lane liveness from
        // a stage-entry invalidity guard. Later stages may begin with local
        // control-flow shortcuts that skip stage work but must preserve the
        // caller-owned active mask for downstream stages.
        if (FinalStages.empty()) {
          errs() << "deepFission: collecting first-stage entry semantics for "
                 << WorkingStage->getName() << "\n";
          Info.Stage0EntrySemantics = collectStage0EntrySemanticInfo(WorkingStage);
          errs() << "deepFission: first-stage entry semantics result func="
                 << WorkingStage->getName()
                 << " present=" << (Info.Stage0EntrySemantics ? "yes" : "no")
                 << "\n";
        } else {
          errs() << "deepFission: skipping non-first-stage entry semantics for "
                 << WorkingStage->getName()
                 << " reason=global_lane_liveness_already_established\n";
        }
        FinalStages.push_back(std::move(Info));
      }
      if (SeedFailed) {
        rollbackSeedArtifacts(M, Seed, WorkingStages, FinalStages);
        continue;
      }

      for (unsigned StageIndex = 0; StageIndex < FinalStages.size();
           ++StageIndex) {
        FinalStages[StageIndex].StageFunc =
            createFinalStageFunction(M, Seed, FinalStages[StageIndex],
                                     StageIndex, FinalStages);
        if (!FinalStages[StageIndex].StageFunc) {
          SeedFailed = true;
          break;
        }
      }
      if (SeedFailed) {
        rollbackSeedArtifacts(M, Seed, WorkingStages, FinalStages);
        continue;
      }

      for (const FinalStageInfo &Info : FinalStages) {
        errs() << "deepFission: [DF][FINAL] event=module.final_validate func="
               << (Info.StageFunc ? Info.StageFunc->getName()
                                  : StringRef("<null>"))
               << " stage="
               << (Info.StageFunc ? getStageIndexForLog(Info.StageFunc)
                                  : std::string("na"))
               << "\n";
        if (!validateFinalStageSemantics(Info.StageFunc)) {
          SeedFailed = true;
          break;
        }
      }
      if (SeedFailed) {
        rollbackSeedArtifacts(M, Seed, WorkingStages, FinalStages);
        continue;
      }

      SmallPtrSet<Function *, 8> CallerFuncs;
      for (CallInst *CallI : OriginalCallsites) {
        if (!CallI || !CallI->getParent())
          continue;
        CallerFuncs.insert(CallI->getFunction());
        if (!rewriteCallerForStageSequence(CallI, Seed, FinalStages)) {
          SeedFailed = true;
          break;
        }
      }
      if (SeedFailed) {
        rollbackSeedArtifacts(M, Seed, WorkingStages, FinalStages);
        continue;
      }

      for (Function *CallerF : CallerFuncs) {
        if (!verifyFunctionOrReport("final deep-fission caller rewrite", CallerF)) {
          SeedFailed = true;
          break;
        }
      }
      if (SeedFailed) {
        rollbackSeedArtifacts(M, Seed, WorkingStages, FinalStages);
        continue;
      }

      SmallPtrSet<GlobalVariable *, 16> SpillGlobalsToErase;
      for (const FinalStageInfo &Info : FinalStages) {
        for (GlobalVariable *GV : Info.UsedSpills)
          SpillGlobalsToErase.insert(GV);
      }

      for (Function *WorkingStage : WorkingStages) {
        if (WorkingStage && WorkingStage->use_empty())
          WorkingStage->eraseFromParent();
      }
      for (GlobalVariable *GV : SpillGlobalsToErase) {
        if (GV && GV->use_empty())
          GV->eraseFromParent();
      }
      if (Seed->use_empty())
        Seed->eraseFromParent();

      if (!SeedChanged)
        errs() << "deepFission: seed " << SeedName
               << " required no structural changes beyond final stage cloning\n";
    }
    return Changed;
  }
};
} // namespace

char DeepFission::ID = 0;
static RegisterPass<DeepFission>
    X("deep-fission", "deep fission on merge-kernel output", false, false);
