#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"


#include <set>
#include <stack>
#include <map>
#include <string>
#include <algorithm>
#include "IDMap.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Operator.h"

// YEBIN: added libs
#include "llvm/IR/IRBuilder.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {
class KernelProfile{
  public:
  std::vector<Value*> loopDims;
  std::map<Value*, int> dim2classify; //0: no meaning 1:outermost loop of grid 2:outermost loop of block
  int gridLoopCnt = 0;
  int blockLoopCnt = 0;
  BasicBlock *kernelBB = nullptr;
  Function *deviceKernel = nullptr;
  CallInst *kernelCall = nullptr;
  Function *newFunc = nullptr;
};

struct MergeKernel : public ModulePass {
  static char ID;
  std::set<Function*>funcs2delete;
  std::map<Function*, Function*> device2newFunc;
  std::map<Function*, std::set<Instruction*>>syncInsts;
  std::set<Function*>syncFuncs;
  std::map<Function*, std::set<CallInst*>>kernelCalls;
  int mdID = 0;
  MergeKernel() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  // Replace inlined CUDA device log function (_ZL3logd) with a simple log() call
  // CUDA inlines math functions like log() into a complex sequence of blocks with
  // NVVM intrinsics. This function simplifies the IR by:
  // 1. Finding ALL _ZL3logd.exit blocks and their corresponding entry blocks
  // 2. Creating standard library log() calls with the original inputs
  // 3. Redirecting control flow to bypass each inlined implementation
  // 4. Deleting the now-unreachable intermediate blocks AFTER all patterns are processed
  void replaceInlinedLog(Function &F, std::vector<Instruction*> &insts2Remove) {
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();
    
    // Structure to hold information about each log pattern
    struct LogPattern {
      BasicBlock *entryBB;
      BasicBlock *exitBB;
      AllocaInst *allocaInst;  // The a.addr.i alloca for this pattern
      Value *logInput;
      Instruction *insertPoint;
      PHINode *resultPhi;
    };
    
    std::vector<LogPattern> patterns;
    
    // First, find ALL a.addr.i allocas and their corresponding entry blocks
    std::map<AllocaInst*, BasicBlock*> allocaToEntryBB;
    std::map<AllocaInst*, Value*> allocaToLogInput;
    std::map<AllocaInst*, Instruction*> allocaToInsertPoint;
    
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
          if (AllocaInst *AI = dyn_cast<AllocaInst>(SI->getPointerOperand())) {
            if (AI->getName().contains("a.addr.i")) {
              // Check if we already have a log call for this alloca in this block
              bool hasLogCall = false;
              for (auto &BI : BB) {
                if (CallInst *CI = dyn_cast<CallInst>(&BI)) {
                  if (Function *calledFunc = CI->getCalledFunction()) {
                    if (calledFunc->getName() == "log") {
                      hasLogCall = true;
                      break;
                    }
                  }
                }
              }
              
              if (!hasLogCall && allocaToEntryBB.find(AI) == allocaToEntryBB.end()) {
                allocaToEntryBB[AI] = &BB;
                allocaToLogInput[AI] = SI->getValueOperand();
                allocaToInsertPoint[AI] = SI->getNextNode();
                errs() << "ANDREW: mergeKernel: found log entry for alloca " << AI->getName() 
                       << " in block " << BB.getName() << "\n";
              }
            }
          }
        }
      }
    }
    
    // Find ALL _ZL3logd.exit blocks
    std::vector<BasicBlock*> exitBlocks;
    for (auto &BB : F) {
      if (BB.getName().contains("_ZL3logd.exit")) {
        exitBlocks.push_back(&BB);
        errs() << "ANDREW: mergeKernel: found exit block: " << BB.getName() << "\n";
      }
    }
    
    if (allocaToEntryBB.empty() || exitBlocks.empty()) {
      return;  // No log patterns found
    }
    
    // Match entry blocks to exit blocks
    // For each entry block, find the exit block that is reachable from it
    for (auto &pair : allocaToEntryBB) {
      AllocaInst *AI = pair.first;
      BasicBlock *entryBB = pair.second;
      
      // Find the exit block reachable from this entry
      // The exit block should be a successor (possibly indirect) of the entry
      BasicBlock *matchedExit = nullptr;
      for (BasicBlock *exitBB : exitBlocks) {
        // Check if this exit is reachable from entry
        std::set<BasicBlock*> visited;
        std::vector<BasicBlock*> worklist;
        worklist.push_back(entryBB);
        bool found = false;
        
        while (!worklist.empty() && !found) {
          BasicBlock *curr = worklist.back();
          worklist.pop_back();
          if (visited.count(curr)) continue;
          visited.insert(curr);
          
          // Only match this exit if it's NOT the entry block itself
          if (curr == exitBB && curr != entryBB) {
            found = true;
            matchedExit = exitBB;
            break;
          }
          
          for (auto *succ : successors(curr)) {
            // Add successor to worklist for further exploration
            if (!visited.count(succ)) {
              worklist.push_back(succ);
            }
            // Check if this successor is the exit we're looking for
            if (succ == exitBB && succ != entryBB) {
              found = true;
              matchedExit = exitBB;
              break;
            }
          }
        }
        
        if (found) break;
      }
      
      if (!matchedExit) {
        // If we couldn't find a matching exit, skip this pattern
        // This can happen for patterns where the entry is in an exit block of another log
        errs() << "ANDREW: mergeKernel: could not find matching exit for alloca " 
               << AI->getName() << ", skipping\n";
        continue;
      }
      
      // Skip patterns where entry == exit (invalid pattern)
      if (matchedExit == entryBB) {
        errs() << "ANDREW: mergeKernel: entry == exit for alloca " 
               << AI->getName() << ", skipping invalid pattern\n";
        continue;
      }
      
      // Get the result phi from the exit block (must be double type to match log() return)
      PHINode *resultPhi = nullptr;
      Type *doubleTy = Type::getDoubleTy(M->getContext());
      for (auto &I : *matchedExit) {
        if (PHINode *phi = dyn_cast<PHINode>(&I)) {
          // Only use a phi that returns double (matching log's return type)
          if (phi->getType() == doubleTy) {
            resultPhi = phi;
            break;
          }
        }
      }
      
      LogPattern pattern;
      pattern.entryBB = entryBB;
      pattern.exitBB = matchedExit;
      pattern.allocaInst = AI;
      pattern.logInput = allocaToLogInput[AI];
      pattern.insertPoint = allocaToInsertPoint[AI];
      pattern.resultPhi = resultPhi;
      patterns.push_back(pattern);
      
      errs() << "ANDREW: mergeKernel: matched pattern - entry: " << entryBB->getName()
             << ", exit: " << matchedExit->getName() 
             << ", alloca: " << AI->getName() << "\n";
    }
    
    if (patterns.empty()) {
      errs() << "ANDREW: mergeKernel: no valid log patterns found\n";
      return;
    }
    
    errs() << "ANDREW: mergeKernel: found " << patterns.size() << " inlined log pattern(s)\n";
    
    // Create the log function type
    Type *doubleTy = Type::getDoubleTy(Ctx);
    FunctionType *logFuncTy = FunctionType::get(doubleTy, {doubleTy}, false);
    FunctionCallee logFunc = M->getOrInsertFunction("log", logFuncTy);
    
    // Process each pattern
    for (auto &pattern : patterns) {
      errs() << "ANDREW: mergeKernel: processing pattern for alloca " 
             << pattern.allocaInst->getName() << "\n";
      errs() << "ANDREW: mergeKernel: log input: " << *pattern.logInput << "\n";
      
      // Create the log call
      CallInst *logCall = CallInst::Create(logFunc, {pattern.logInput}, "log_result", 
                                           pattern.insertPoint);
      errs() << "ANDREW: mergeKernel: created log call: " << *logCall << "\n";
      
      // Replace uses of the result phi with the log call
      if (pattern.resultPhi) {
        pattern.resultPhi->replaceAllUsesWith(logCall);
        pattern.resultPhi->eraseFromParent();
        errs() << "ANDREW: mergeKernel: deleted result phi node\n";
      }
      
      // Redirect control flow: entry block should branch directly to exit block
      Instruction *oldTerm = pattern.entryBB->getTerminator();
      BranchInst::Create(pattern.exitBB, oldTerm);
      oldTerm->eraseFromParent();
      
      errs() << "ANDREW: mergeKernel: redirected control flow from " 
             << pattern.entryBB->getName() << " to " << pattern.exitBB->getName() << "\n";
    }
    
    // NOW delete all unreachable blocks (after ALL patterns are processed)
    std::set<BasicBlock*> reachable;
    std::vector<BasicBlock*> worklist;
    worklist.push_back(&F.getEntryBlock());
    
    while (!worklist.empty()) {
      BasicBlock *BB = worklist.back();
      worklist.pop_back();
      if (reachable.count(BB)) continue;
      reachable.insert(BB);
      for (auto *Succ : successors(BB)) {
        worklist.push_back(Succ);
      }
    }
    
    // Collect unreachable blocks
    std::vector<BasicBlock*> toDelete;
    for (auto &BB : F) {
      if (!reachable.count(&BB)) {
        toDelete.push_back(&BB);
      }
    }
    
    errs() << "ANDREW: mergeKernel: found " << toDelete.size() << " unreachable blocks to delete\n";
    
    // Delete unreachable blocks
    for (auto *BB : toDelete) {
      BB->dropAllReferences();
    }
    for (auto *BB : toDelete) {
      BB->eraseFromParent();
    }
    
    errs() << "ANDREW: mergeKernel: replaced " << patterns.size() 
           << " inlined log(s) with log() calls and cleaned up dead blocks\n";
  }

  // Replace inlined CUDA device exp function (_ZL3expd) with a simple exp() call.
  // Similar to replaceInlinedLog, this:
  // 1. Finds stores into a.addr.i (the exp input capture point)
  // 2. Matches reachable _ZL3expd.exit blocks
  // 3. Creates exp() calls and rewires control-flow to bypass inlined exp IR
  // 4. Deletes blocks that become unreachable
  void replaceInlinedExp(Function &F, std::vector<Instruction*> &insts2Remove) {
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    struct ExpPattern {
      BasicBlock *entryBB;
      BasicBlock *exitBB;
      AllocaInst *allocaInst;
      Value *expInput;
      Instruction *insertPoint;
      PHINode *resultPhi;
    };

    std::vector<ExpPattern> patterns;
    std::map<AllocaInst*, BasicBlock*> allocaToEntryBB;
    std::map<AllocaInst*, Value*> allocaToExpInput;
    std::map<AllocaInst*, Instruction*> allocaToInsertPoint;

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
          if (AllocaInst *AI = dyn_cast<AllocaInst>(SI->getPointerOperand())) {
            if (AI->getName().contains("a.addr.i")) {
              bool hasExpCall = false;
              for (auto &BI : BB) {
                if (CallInst *CI = dyn_cast<CallInst>(&BI)) {
                  if (Function *calledFunc = CI->getCalledFunction()) {
                    if (calledFunc->getName() == "exp") {
                      hasExpCall = true;
                      break;
                    }
                  }
                }
              }

              if (!hasExpCall && allocaToEntryBB.find(AI) == allocaToEntryBB.end()) {
                allocaToEntryBB[AI] = &BB;
                allocaToExpInput[AI] = SI->getValueOperand();
                allocaToInsertPoint[AI] = SI->getNextNode();
                errs() << "ANDREW: mergeKernel: found exp entry for alloca " << AI->getName()
                       << " in block " << BB.getName() << "\n";
              }
            }
          }
        }
      }
    }

    std::vector<BasicBlock*> exitBlocks;
    for (auto &BB : F) {
      if (BB.getName().contains("_ZL3expd.exit")) {
        exitBlocks.push_back(&BB);
        errs() << "ANDREW: mergeKernel: found exp exit block: " << BB.getName() << "\n";
      }
    }

    if (allocaToEntryBB.empty() || exitBlocks.empty()) {
      errs() << "ANDREW: mergeKernel: no exp patterns found in function " << F.getName() << "\n";
      return;
    }

    for (auto &pair : allocaToEntryBB) {
      AllocaInst *AI = pair.first;
      BasicBlock *entryBB = pair.second;

      BasicBlock *matchedExit = nullptr;
      for (BasicBlock *exitBB : exitBlocks) {
        std::set<BasicBlock*> visited;
        std::vector<BasicBlock*> worklist;
        worklist.push_back(entryBB);
        bool found = false;

        while (!worklist.empty() && !found) {
          BasicBlock *curr = worklist.back();
          worklist.pop_back();
          if (visited.count(curr)) continue;
          visited.insert(curr);

          if (curr == exitBB && curr != entryBB) {
            found = true;
            matchedExit = exitBB;
            break;
          }

          for (auto *succ : successors(curr)) {
            if (!visited.count(succ))
              worklist.push_back(succ);
            if (succ == exitBB && succ != entryBB) {
              found = true;
              matchedExit = exitBB;
              break;
            }
          }
        }

        if (found)
          break;
      }

      if (!matchedExit) {
        errs() << "ANDREW: mergeKernel: could not find matching exp exit for alloca "
               << AI->getName() << ", skipping\n";
        continue;
      }

      if (matchedExit == entryBB) {
        errs() << "ANDREW: mergeKernel: exp entry == exit for alloca "
               << AI->getName() << ", skipping invalid pattern\n";
        continue;
      }

      PHINode *resultPhi = nullptr;
      Type *doubleTy = Type::getDoubleTy(M->getContext());
      for (auto &I : *matchedExit) {
        if (PHINode *phi = dyn_cast<PHINode>(&I)) {
          if (phi->getType() == doubleTy) {
            resultPhi = phi;
            break;
          }
        }
      }

      if (!resultPhi) {
        errs() << "ANDREW: mergeKernel: exp exit block " << matchedExit->getName()
               << " has no double phi result, skipping\n";
        continue;
      }

      ExpPattern pattern;
      pattern.entryBB = entryBB;
      pattern.exitBB = matchedExit;
      pattern.allocaInst = AI;
      pattern.expInput = allocaToExpInput[AI];
      pattern.insertPoint = allocaToInsertPoint[AI];
      pattern.resultPhi = resultPhi;
      patterns.push_back(pattern);

      errs() << "ANDREW: mergeKernel: matched exp pattern - entry: " << entryBB->getName()
             << ", exit: " << matchedExit->getName()
             << ", alloca: " << AI->getName() << "\n";
    }

    if (patterns.empty()) {
      errs() << "ANDREW: mergeKernel: no valid exp patterns found\n";
      return;
    }

    errs() << "ANDREW: mergeKernel: found " << patterns.size() << " inlined exp pattern(s)\n";

    Type *doubleTy = Type::getDoubleTy(Ctx);
    FunctionType *expFuncTy = FunctionType::get(doubleTy, {doubleTy}, false);
    FunctionCallee expFunc = M->getOrInsertFunction("exp", expFuncTy);

    for (auto &pattern : patterns) {
      errs() << "ANDREW: mergeKernel: processing exp pattern for alloca "
             << pattern.allocaInst->getName() << "\n";
      errs() << "ANDREW: mergeKernel: exp input: " << *pattern.expInput << "\n";

      CallInst *expCall = CallInst::Create(expFunc, {pattern.expInput}, "exp_result",
                                           pattern.insertPoint);
      errs() << "ANDREW: mergeKernel: created exp call: " << *expCall << "\n";

      pattern.resultPhi->replaceAllUsesWith(expCall);
      pattern.resultPhi->eraseFromParent();
      errs() << "ANDREW: mergeKernel: deleted exp result phi node\n";

      Instruction *oldTerm = pattern.entryBB->getTerminator();
      BranchInst::Create(pattern.exitBB, oldTerm);
      oldTerm->eraseFromParent();
      errs() << "ANDREW: mergeKernel: redirected exp control flow from "
             << pattern.entryBB->getName() << " to " << pattern.exitBB->getName() << "\n";
    }

    std::set<BasicBlock*> reachable;
    std::vector<BasicBlock*> worklist;
    worklist.push_back(&F.getEntryBlock());

    while (!worklist.empty()) {
      BasicBlock *BB = worklist.back();
      worklist.pop_back();
      if (reachable.count(BB)) continue;
      reachable.insert(BB);
      for (auto *Succ : successors(BB)) {
        worklist.push_back(Succ);
      }
    }

    std::vector<BasicBlock*> toDelete;
    for (auto &BB : F) {
      if (!reachable.count(&BB)) {
        toDelete.push_back(&BB);
      }
    }

    errs() << "ANDREW: mergeKernel: found " << toDelete.size()
           << " unreachable blocks after exp lifting\n";

    for (auto *BB : toDelete) {
      BB->dropAllReferences();
    }
    for (auto *BB : toDelete) {
      BB->eraseFromParent();
    }

    errs() << "ANDREW: mergeKernel: replaced " << patterns.size()
           << " inlined exp(s) with exp() calls and cleaned up dead blocks\n";
  }

  void findThreadDim(KernelProfile *kernelProfile, Function &F, LoadInst *DimArg, bool isBlockDim){
    //LoadInst *DimArg = dyn_cast<LoadInst>(CI->getArgOperand(2));
    assert(DimArg && "mergeKernel: DimArg is not a load inst\n");
    GetElementPtrInst *ArgGep = dyn_cast<GetElementPtrInst>(DimArg->getOperand(0));
    assert(ArgGep && "mergeKernel: DimGep is not a gep inst\n");
    AllocaInst *Coerce = dyn_cast<AllocaInst>(ArgGep->getOperand(0));
    assert(Coerce && "mergeKernel: DimAllocaCoerce is not an alloca inst\n");
    BitCastInst *CoerceBitCast = nullptr;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      BitCastInst *bitcast = dyn_cast<BitCastInst>(&*I);
      if(!bitcast) continue;
      if(bitcast->getOperand(0) != Coerce) continue;
      CoerceBitCast = bitcast;
      break;
    }
    assert(CoerceBitCast && "mergeKernel: CoerceBitCast is not a bit cast inst\n");
    BitCastInst *AggBitCast = nullptr;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(&*I);
      if(!CI) continue;
      Function *called = CI->getCalledFunction();
      if(!called) continue;
      if(called->getName() != "llvm.memcpy.p0i8.p0i8.i64") continue;
      if(CI->getArgOperand(0) != CoerceBitCast) continue;
      BitCastInst *bitcast = dyn_cast<BitCastInst>(CI->getArgOperand(1));
      AggBitCast = bitcast;
      break;
    }
    assert(AggBitCast && "mergeKernel: AggMemcpy not found \n");
    AllocaInst *AggAlloca = dyn_cast<AllocaInst>(AggBitCast->getOperand(0));
    assert(AggAlloca && "mergeKernel: gridDimAlloca is not an alloca inst \n");
    bool foundDim = false;
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(&*I);
      if(!CI) continue;
      Function *called = CI->getCalledFunction();
      if(!called) continue;
      if(!called->getName().contains("_ZN4dim3C2Ejjj")) continue;
      if(CI->getArgOperand(0) != AggAlloca) continue;

      bool classifiedPrimaryDim = false;
      for(int i=1; i<=3; ++i){
        Value *dim = CI->getArgOperand(i);
        bool isOne = false;
        if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
          if(constInt->getSExtValue() == 1)
            isOne = true;
        kernelProfile->loopDims.push_back(dim);
        if(isBlockDim){
          if(!isOne) kernelProfile->blockLoopCnt ++;
          if(!isOne && !classifiedPrimaryDim){
            kernelProfile->dim2classify[dim] = 2;
            classifiedPrimaryDim = true;
          } else {
            kernelProfile->dim2classify[dim] = 0;
          }
        }
        else{
          if(!isOne) kernelProfile->gridLoopCnt ++;
          if(!isOne && !classifiedPrimaryDim){
            kernelProfile->dim2classify[dim] = 1;
            classifiedPrimaryDim = true;
          } else {
            kernelProfile->dim2classify[dim] = 0;
          }
        }

        errs() << "mergeKernel: Dim " << i << " : " << *dim <<"\n";
      }


      foundDim = true;
      break;
    }
    if(!foundDim){
      BitCastInst *AggBitCast2 = nullptr;
      for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
        BitCastInst *bitcast = dyn_cast<BitCastInst>(&*I);
        if(!bitcast) continue;
        if(bitcast->getOperand(0) != AggAlloca) continue;
        AggBitCast2 = bitcast;
        break;
      }
      assert(AggBitCast2 && "mergeKernel: AggBitCast2 is not found \n");
      BitCastInst *DimBitCast = nullptr;
      for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
        CallInst *CI = dyn_cast<CallInst>(&*I);
        if(!CI) continue;
        Function *called = CI->getCalledFunction();
        if(!called) continue;
        if(called->getName() != "llvm.memcpy.p0i8.p0i8.i64") continue;
        if(CI->getArgOperand(0) != AggBitCast2) continue;
        BitCastInst *bitcast = dyn_cast<BitCastInst>(CI->getArgOperand(1));
        DimBitCast = bitcast;
        break;
      }
      AllocaInst *DimAlloca = dyn_cast<AllocaInst>(DimBitCast->getOperand(0));
      assert(DimAlloca && "mergeKernel: DimAlloca is not found \n");
      for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
        CallInst *CI = dyn_cast<CallInst>(&*I);
        if(!CI) continue;
        Function *called = CI->getCalledFunction();
        if(!called) continue;
        if(!called->getName().contains("_ZN4dim3C2Ejjj")) continue;
        if(CI->getArgOperand(0) != DimAlloca) continue;
        bool classifiedPrimaryDim = false;
        for(int i=1; i<=3; ++i){
          Value *dim = CI->getArgOperand(i);
          bool isOne = false;
          if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
            if(constInt->getSExtValue() == 1)
              isOne = true;
          kernelProfile->loopDims.push_back(dim);
          if(isBlockDim){
            if(!isOne) kernelProfile->blockLoopCnt ++;
            if(!isOne && !classifiedPrimaryDim){
              kernelProfile->dim2classify[dim] = 2;
              classifiedPrimaryDim = true;
            } else {
              kernelProfile->dim2classify[dim] = 0;
            }
          }
          else{
            if(!isOne) kernelProfile->gridLoopCnt ++;
            if(!isOne && !classifiedPrimaryDim){
              kernelProfile->dim2classify[dim] = 1;
              classifiedPrimaryDim = true;
            } else {
              kernelProfile->dim2classify[dim] = 0;
            }
          }
        }
        foundDim = true;
        break;
      }
    }
  }

  void splitFunction(LLVMContext &Context, Function* F, std::set<Instruction*> insts) {
    IRBuilder<> Builder(Context);
    unsigned barrierCount = insts.size();
    // Convert to std::string and remove null bytes to prevent assertion failures
    std::string orignalName = F->getName().str();
    orignalName.erase(std::remove(orignalName.begin(), orignalName.end(), '\0'), orignalName.end());
    F->setName(orignalName+"0");

    // Order needs to be later instructions first
    // Is it preserved?
    for(auto it = insts.rbegin(); it != insts.rend(); it++) {
      Instruction *syncInst = *it;
      auto *BB = syncInst->getParent();
      auto *splitPoint = BB->splitBasicBlock(syncInst, "syncpoint."+std::to_string(barrierCount));
      auto* LI = &getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
      // create new kernel function
      auto *newFunc = Function::Create(
            F->getFunctionType(),
            F->getLinkage(),
            orignalName+std::to_string(barrierCount--),
            F->getParent()
          );
      //copy old kernel over to the new
      ValueToValueMapTy VMap;
      auto NewFArgIt = newFunc->arg_begin();
      for (auto &Arg: F->args()) {
        auto ArgName = Arg.getName();
        NewFArgIt->setName(ArgName);
        VMap[&Arg] = &(*NewFArgIt++);
      }

      SmallVector<ReturnInst*, 8> Returns;
      llvm::CloneFunctionInto(newFunc, F, VMap, false, Returns);
      auto *newBB = cast<Instruction>(*VMap[BB->getTerminator()]).getParent();
      auto *newSplitBB = cast<Instruction>(*VMap[splitPoint->getTerminator()]).getParent();

      // call newly split function
      for(auto *callInst: kernelCalls[F]) {
        Builder.SetInsertPoint(callInst->getNextNode());
        std::vector<Value*> args;
        for(auto &arg: callInst->args()) args.push_back(arg);
        Builder.CreateCall(newFunc, args);
      }

      // prune function body based on barrier
      // FIXME: currently assumes the synchronization is not within a loop
      assert(!LI->getLoopFor(BB) && "The synchronization point is in a loop!!!\n");

      // all predecessors of splitPoint (not inclusive) are part of prevF
      // others are part of splitF
      Builder.SetInsertPoint(BB->getTerminator());
      Builder.CreateRetVoid();
      BB->getTerminator()->eraseFromParent();

      std::set<BasicBlock*> prevEraseList;
      std::set<BasicBlock*> splitEraseList;
      // workaround for use-def errors
      for(df_iterator<BasicBlock*> SI = df_begin(splitPoint); SI != df_end(splitPoint); ++SI) {
        BasicBlock* succ = *SI;
        prevEraseList.insert(succ);
      }
      for(auto &bb: prevEraseList) {
        Builder.SetInsertPoint(bb->getTerminator());
        Builder.CreateRetVoid();
        bb->getTerminator()->eraseFromParent();
      }
      for(auto &bb: prevEraseList) bb->eraseFromParent();

      // All successors of splitPoint (inclusive) are part of splitF
      // Previous instruction must also be considered
      // FIXME: assumes all "setup" insts are in the entry block
      // TODO: Live-in Live-out analyses, modify function signature
      for(idf_iterator<BasicBlock*> PI = idf_begin(newBB); PI != idf_end(newBB); ++PI) {
        BasicBlock* pred = *PI;
        if (pred != &newFunc->getEntryBlock())
          splitEraseList.insert(pred);
      }
      // handle entry block seperately
      Builder.SetInsertPoint(newFunc->getEntryBlock().getTerminator());
      Builder.CreateBr(newSplitBB);
      newFunc->getEntryBlock().getTerminator()->eraseFromParent();
      // same as prevF
      for(auto &bb: splitEraseList) {
        Builder.SetInsertPoint(bb->getTerminator());
        Builder.CreateRetVoid();
        bb->getTerminator()->eraseFromParent();
      }
      for(auto &bb: splitEraseList) bb->eraseFromParent();
    }
  }

  // Copied from LoopSimplify pass
  BasicBlock* makeLoopPreheader(LoopInfo& LI, DominatorTree& DT, Loop* L) {
    BasicBlock *header = L->getHeader();

    std::vector<BasicBlock*> enteringBBs;
    for(pred_iterator PI = pred_begin(header); PI != pred_end(header); ++PI) {
      if(!L->contains(*PI))
        enteringBBs.push_back(*PI);
    }

    BasicBlock *preheader = SplitBlockPredecessors(header, enteringBBs, ".preheader");

    if(Loop* parent = L->getParentLoop())
      parent->addBasicBlockToLoop(preheader, LI);

    DT.splitBlock(preheader);

    return preheader;
  }

  void splitLoop(LLVMContext &Context, CallInst *I, unsigned cloneNum) {
    IRBuilder<> Builder(Context);
    // Duplicate the kernel call loop
    // FIXME: Assume a 2d grid for now...
    auto *F = I->getFunction();
    auto &LI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
    auto threadLoop = LI.getLoopFor(I->getParent());
    // insert preheader if it does not exist
    if(!threadLoop->getLoopPreheader()) {
      BasicBlock *preheader = makeLoopPreheader(LI, DT, threadLoop);
      errs() << threadLoop->getLoopPreheader()->getName() << "\n";
    }
    // assume single predecessor of preheader (one entrance from outer loop)
    auto *entering = threadLoop->getLoopPreheader()->getSinglePredecessor();
    assert(entering && "Not a single predecessor of preheader!!!\n");

    Loop* prevLoop = threadLoop;
    for(unsigned i = 0; i < cloneNum; i++) {
      errs() << "CLONING " << i << "\n"; 
      ValueToValueMapTy VMap;
      SmallVector<BasicBlock* , 4> newBlocks;
      auto *newLoop = cloneLoopWithPreheader(prevLoop->getLoopPreheader(), &F->getEntryBlock(), threadLoop, VMap, ".clone"+std::to_string(i), &LI, &DT, newBlocks);
      remapInstructionsInBlocks(newBlocks, VMap);
      entering->getTerminator()->replaceUsesOfWith(prevLoop->getLoopPreheader(), newLoop->getLoopPreheader());
      newLoop->getHeader()->getTerminator()->replaceUsesOfWith(threadLoop->getExitBlock(), prevLoop->getLoopPreheader());
      auto* call = cast<Instruction>(VMap[I]);
      for(unsigned j = 0; j <= cloneNum; j++) {
        auto* tempCall = call;
        call = call->getNextNode();
        errs() << "IDX: " << i+j << "\n";
        if(j+i != cloneNum - 1)
          tempCall->eraseFromParent();
      }

      prevLoop = newLoop;
    }
    // Original loop, now last one
    for(unsigned j = 0; j < cloneNum; j++) {
      auto* tempI = I;
      CallInst* CI = dyn_cast<CallInst>(I->getNextNode());
      assert(CI && "Next inst is not a CallInst!!\n");
      I = CI;
      tempI->eraseFromParent();
    }
  }

  bool runOnModule(Module &M) override {
    std::set<Function*> atomicImplFuncs;
    // Demangle mangled global variable names so CBE emits readable symbols
    // (e.g. _ZL11grid_points -> grid_points).
    auto sanitizeIdentifier = [](const std::string &name) -> std::string {
      std::string out;
      out.reserve(name.size());
      for (char ch : name) {
        bool isAlphaNumUnderscore =
            ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
             (ch >= '0' && ch <= '9') || ch == '_');
        out.push_back(isAlphaNumUnderscore ? ch : '_');
      }
      if (out.empty())
        return out;
      if (out[0] >= '0' && out[0] <= '9')
        out = "g_" + out;
      return out;
    };

    errs() << "mergeKernel: starting global demangle pass\n";
    std::set<std::string> usedGlobalNames;
    unsigned globalsVisited = 0;
    unsigned globalsRenamed = 0;
    for (auto &GV : M.globals())
      usedGlobalNames.insert(GV.getName().str());

    for (auto &GV : M.globals()) {
      ++globalsVisited;
      StringRef oldNameRef = GV.getName();
      // Keep frontend-generated C string constants stable.
      if (oldNameRef.startswith(".str")) {
        errs() << "mergeKernel: demangle skip (string const): " << oldNameRef << "\n";
        continue;
      }
      if (!oldNameRef.startswith("_Z")) {
        errs() << "mergeKernel: demangle skip (not mangled): " << oldNameRef << "\n";
        continue;
      }

      errs() << "mergeKernel: demangle candidate: " << oldNameRef << "\n";

      std::string demangled = demangle(oldNameRef.str());
      if (demangled.empty() || demangled == oldNameRef.str()) {
        errs() << "mergeKernel: demangle failed/unchanged: " << oldNameRef << "\n";
        continue;
      }

      size_t parenPos = demangled.find('(');
      if (parenPos != std::string::npos)
        demangled = demangled.substr(0, parenPos);
      size_t nsPos = demangled.rfind("::");
      if (nsPos != std::string::npos && nsPos + 2 < demangled.size())
        demangled = demangled.substr(nsPos + 2);

      std::string baseName = sanitizeIdentifier(demangled);
      if (baseName.empty() || baseName == oldNameRef.str()) {
        errs() << "mergeKernel: demangle skip (invalid/same sanitized): " << oldNameRef
               << " -> " << baseName << "\n";
        continue;
      }

      std::string candidate = baseName;
      int suffix = 0;
      while ((M.getNamedValue(candidate) != nullptr &&
              M.getNamedValue(candidate) != &GV) ||
             usedGlobalNames.count(candidate) != 0) {
        errs() << "mergeKernel: demangle collision for " << oldNameRef
               << " at candidate " << candidate << ", retrying\n";
        candidate = baseName + "_" + std::to_string(++suffix);
      }

      if (candidate != oldNameRef.str()) {
        errs() << "mergeKernel: demangled global " << oldNameRef
               << " -> " << candidate << "\n";
        GV.setName(candidate);
        ++globalsRenamed;
      }
      usedGlobalNames.insert(candidate);
    }
    errs() << "mergeKernel: global demangle pass done, visited=" << globalsVisited
           << ", renamed=" << globalsRenamed << "\n";

    //transform target
    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
      std::map<CallInst*, KernelProfile*> kernelProfiles;
      Function *F = &*FI;
      if(F->isDeclaration()) continue;
      if(F->hasFnAttribute("target-features"))
        F->removeFnAttr("target-features");
      if(F->hasFnAttribute("target-cpu"))
        F->removeFnAttr("target-cpu");
      BasicBlock *kernelBB = nullptr;
      std::vector<Instruction*> insts2Remove;
      
      // Replace inlined math patterns first (before processing intrinsics)
      replaceInlinedLog(*F, insts2Remove);
      replaceInlinedExp(*F, insts2Remove);
      
      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
        if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(&*I)) {
          if (RMW->getOperation() == AtomicRMWInst::Add ||
              RMW->getOperation() == AtomicRMWInst::FAdd) {
            LLVMContext &C = RMW->getContext();
            MDNode *N = MDNode::get(C, MDString::get(C, ""));
            RMW->setMetadata("tulip.atomic.add", N);
            // When lowering atomicAdd(..., +/-1) where the old value is
            // consumed, request an unambiguous OpenMP post-inc/dec capture.
            if (!RMW->user_empty() && !RMW->getType()->isVoidTy()) {
              if (ConstantInt *CI = dyn_cast<ConstantInt>(RMW->getValOperand())) {
                if (CI->isMinusOne()) {
                  MDNode *PostDec = MDNode::get(C, MDString::get(C, ""));
                  RMW->setMetadata("tulip.atomic.capture.postdec", PostDec);
                } else if (CI->isOne()) {
                  MDNode *PostInc = MDNode::get(C, MDString::get(C, ""));
                  RMW->setMetadata("tulip.atomic.capture.postinc", PostInc);
                }
              }
            }
          }
        }
        if(CallInst *CI = dyn_cast<CallInst>(&*I)){
          Function* calledFunc = CI->getCalledFunction();
          if (!calledFunc) continue; // Skip indirect calls and inline assembly
          // Preserve OpenMP runtime orchestration as-is. C backend expects
          // canonical __kmpc_* calls to reconstruct parallel regions.
          if (calledFunc->getName().startswith("__kmpc_") ||
              calledFunc->getName().startswith("omp_")) {
            errs() << "mergeKernel: preserving OpenMP runtime call: " << *CI << "\n";
            continue;
          }
          if (calledFunc->getName().contains("atomicAdd")) {
            LLVMContext &C = CI->getContext();
            MDNode *N = MDNode::get(C, MDString::get(C, ""));
            CI->setMetadata("tulip.atomic.add", N);
            // Same marker for call-based atomicAdd lowering.
            if (!CI->user_empty() && !CI->getType()->isVoidTy() &&
                CI->arg_size() >= 2) {
              if (ConstantInt *Arg1 = dyn_cast<ConstantInt>(CI->getArgOperand(1))) {
                if (Arg1->isMinusOne()) {
                  MDNode *PostDec = MDNode::get(C, MDString::get(C, ""));
                  CI->setMetadata("tulip.atomic.capture.postdec", PostDec);
                } else if (Arg1->isOne()) {
                  MDNode *PostInc = MDNode::get(C, MDString::get(C, ""));
                  CI->setMetadata("tulip.atomic.capture.postinc", PostInc);
                }
              }
            }
            atomicImplFuncs.insert(calledFunc);
            continue;
          }
          if (calledFunc->getName().contains("atomicCAS")) {
            atomicImplFuncs.insert(calledFunc);
          }
          // Collect sync functions
          if(calledFunc->getName().contains("llvm.nvvm.barrier")){
            syncFuncs.insert(calledFunc);
            syncInsts[F].insert(CI);
            //funcs2delete.insert(calledFunc);
            //insts2Remove.push_back(CI);
          }
          //if(calledFunc->getName().contains("checkCudaError"))
          else if (calledFunc->getName().contains("llvm.nvvm.fmax")){
            errs() << "mergeKernel: found nvvm fmax declaration\n";
            auto sqrtFuncTy = calledFunc->getFunctionType();
            Function *F = Function::Create(sqrtFuncTy, Function::ExternalLinkage, "fmax", M);
            CallSite CS(CI);
            SmallVector<Value *, 4> args(CS.arg_begin(), CS.arg_end());
            auto sqrtFunc = F->getParent()->getOrInsertFunction("fmax", sqrtFuncTy);
            auto sqrtCall = CallInst::Create(
                  sqrtFunc,
                  args,
                  "",
                  CI
                );
            for(User *U : CI->users()){
              Instruction *inst = dyn_cast<Instruction>(U);
              if(!inst) continue;
              for (auto OI = inst->op_begin(), OE = inst->op_end(); OI != OE; ++OI){
                Value *val = *OI;
                if(val == CI)
                  *OI = sqrtCall;
              }
            }
            insts2Remove.push_back(CI);
          }
          else if (calledFunc->getName().contains("llvm.nvvm.sqrt")){
            errs() << "mergeKernel: found nvvm sqrt declaration\n";
            auto sqrtFuncTy = calledFunc->getFunctionType();
            Function *F = Function::Create(sqrtFuncTy, Function::ExternalLinkage, "sqrt", M);
            CallSite CS(CI);
            SmallVector<Value *, 4> args(CS.arg_begin(), CS.arg_end());
            auto sqrtFunc = F->getParent()->getOrInsertFunction("sqrt", sqrtFuncTy);
            auto sqrtCall = CallInst::Create(
                  sqrtFunc,
                  args,
                  "",
                  CI
                );
            for(User *U : CI->users()){
              Instruction *inst = dyn_cast<Instruction>(U);
              if(!inst) continue;
              for (auto OI = inst->op_begin(), OE = inst->op_end(); OI != OE; ++OI){
                Value *val = *OI;
                if(val == CI)
                  *OI = sqrtCall;
              }
            }
            insts2Remove.push_back(CI);
          }
          // Note: llvm.nvvm.fma.rn.d is part of inlined _ZL3logd
          // DCE will remove dead fma calls after replaceInlinedLog
          // Replace nvvm fabs.d with standard fabs (used for max(fabs(t3), fabs(t4)) outside log)
          else if (calledFunc->getName().contains("llvm.nvvm.fabs.d")){
            errs() << "mergeKernel: found nvvm fabs.d declaration\n";
            auto fabsFuncTy = calledFunc->getFunctionType();
            CallSite CS(CI);
            SmallVector<Value *, 4> args(CS.arg_begin(), CS.arg_end());
            auto fabsFunc = F->getParent()->getOrInsertFunction("fabs", fabsFuncTy);
            auto fabsCall = CallInst::Create(
                  fabsFunc,
                  args,
                  "",
                  CI
                );
            CI->replaceAllUsesWith(fabsCall);
            insts2Remove.push_back(CI);
          }
          else if(calledFunc->getName().contains("_ZN4dim3C2Ejjj")){
            funcs2delete.insert(calledFunc);
            insts2Remove.push_back(CI);

            Value *dimPtr = CI->getArgOperand(0);
            PointerType *dimPtrTy = dyn_cast<PointerType>(dimPtr->getType());
            assert(dimPtrTy && "dimPtr isn't of pointer type!\n");
            auto i32Ty = Type::getInt32Ty(CI->getContext());
            for(int i=1; i<=3; ++i){
              Value *dim = CI->getArgOperand(i);
              //replace _ZN4dim3C2Ejjj with stores to dim3 struct
              std::vector<Value*>idxList;
              idxList.push_back( ConstantInt::get(i32Ty, 0));
              idxList.push_back( ConstantInt::get(i32Ty, i-1));
              auto gep = GetElementPtrInst::Create(dimPtrTy->getPointerElementType(), dimPtr, idxList, "dim3gep."+std::to_string(i-1), CI);
              //StoreInst (Value *Val, Value *Ptr, Instruction *InsertBefore)
              new StoreInst(dim, gep, CI);
            }
        }
        else if(calledFunc->getName().contains("cudaFree")){
           // auto ptr = CI->getArgOperand(0);
           // CallInst::CreateFree(ptr, CI);
            insts2Remove.push_back(CI);
            funcs2delete.insert(calledFunc);
        }
        else if(calledFunc->getName().contains("cudaConfigureCall")){
            //temporous registers
            CallInst *kernelCall = nullptr;
            Function *deviceKernel = nullptr;
            Function *newFunc = nullptr;

            errs() << "CudaFE: found cudaConfigureCall\n";
            kernelProfiles[CI] = new KernelProfile();
            //remove control flow caused by configuration failure
            BranchInst *CF2Remove = dyn_cast<BranchInst>(CI->getParent()->getTerminator());
            CmpInst *cmp = dyn_cast<CmpInst>(CF2Remove->getCondition());
            if(cmp){
              if(cmp->getPredicate() == CmpInst::ICMP_EQ || cmp->getPredicate() == CmpInst::ICMP_NE){
                auto opnd0 = cmp->getOperand(0);
                auto opnd1 = cmp->getOperand(1);
                bool ConfigureThenBranchPattern = false;
                if(ConstantInt *integer = dyn_cast<ConstantInt>(opnd0)){
                  if(integer->getZExtValue() == 0 && opnd1 == CI && cmp->getPredicate() == CmpInst::ICMP_EQ)
                    ConfigureThenBranchPattern = false;
                  else if (integer->getZExtValue() == 0 && opnd1 == CI && cmp->getPredicate() == CmpInst::ICMP_NE)
                    ConfigureThenBranchPattern = true;
                } else if(ConstantInt *integer = dyn_cast<ConstantInt>(opnd1)){
                  if(integer->getZExtValue() == 0 && opnd0 == CI && cmp->getPredicate() == CmpInst::ICMP_EQ)
                    ConfigureThenBranchPattern = true;
                  else if (integer->getZExtValue() == 0 && opnd0 == CI && cmp->getPredicate() == CmpInst::ICMP_NE)
                    ConfigureThenBranchPattern = false;
                }
                if(ConfigureThenBranchPattern){
                  CF2Remove->setCondition(ConstantInt::get(Type::getInt1Ty(CF2Remove->getContext()), 1));
                  kernelProfiles[CI]->kernelBB = CF2Remove->getSuccessor(0);
                  errs() << "mergeKernel: kernelBB: " << *(kernelProfiles[CI]->kernelBB) << "\n";
                } else {
                  CF2Remove->setCondition(ConstantInt::get(Type::getInt1Ty(CF2Remove->getContext()), 0));
                  kernelProfiles[CI]->kernelBB = CF2Remove->getSuccessor(1);
                  errs() << "mergeKernel: kernelBB: " << *(kernelProfiles[CI]->kernelBB) << "\n";
                }

                //write metadata to the function call
                for(auto &I : *(kernelProfiles[CI]->kernelBB)){
                  if(CallInst *ci = dyn_cast<CallInst>(&I)){
                    Function *calledKernelHost = ci->getCalledFunction();
                    if (!calledKernelHost || calledKernelHost->isDeclaration()) continue;
                    for (inst_iterator I = inst_begin(calledKernelHost), E = inst_end(calledKernelHost); I != E; ++I) {
                      if(CallInst *callInst = dyn_cast<CallInst>(&*I)){
                        Function *calledF = callInst->getCalledFunction();
                        if(!calledF) continue;
                        if(calledF->getName().contains("cudaLaunch")){
                          kernelProfiles[CI]->kernelCall = ci;
                          kernelCall = ci;
                          break;
                        }
                      }
                    }
                  }

                  if(kernelCall)
                    break;
                }

                errs() << "mergeKernel: kernelCall: " << *(kernelCall) << "\n";
                assert(kernelCall && "mergeKernel: din't find kernel call!\n");
                errs() << "mergeKernel: kernel call: " << *(kernelCall) << "\n";
                insts2Remove.push_back(cmp);
                funcs2delete.insert(calledFunc);
            }
          }

            assert(kernelCall && "mergeKernel: din't find kernel call!\n");
            errs() << "mergeKernel: kernel call: " << *(kernelCall) << "\n";

            //TODO: mem2reg sort of changes everything with the pattern matching
            errs() << "MergeKernel: blocks per grid:\n";
            if(isa<ConstantInt>(CI->getArgOperand(0))){
              Value *dim = CI->getArgOperand(0);
              kernelProfiles[CI]->loopDims.push_back(dim);
              kernelProfiles[CI]->dim2classify[dim] = 1;
              bool isOne = false;
              if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
                if(constInt->getSExtValue() == 1)
                  isOne = true;
              if(!isOne) kernelProfiles[CI]->gridLoopCnt ++;
            }
            else{
              findThreadDim(kernelProfiles[CI], *F, dyn_cast<LoadInst>(CI->getArgOperand(0)), false);
            }

            errs() << "MergeKernel: threads per block:\n";
            if(isa<ConstantInt>(CI->getArgOperand(2))){
              Value *dim = CI->getArgOperand(2);
              kernelProfiles[CI]->loopDims.push_back(dim);
              kernelProfiles[CI]->dim2classify[dim] = 2;
              bool isOne = false;
              if(ConstantInt *constInt = dyn_cast<ConstantInt>(dim))
                if(constInt->getSExtValue() == 1)
                  isOne = true;
              if(!isOne) kernelProfiles[CI]->blockLoopCnt ++;
            }
            else{
              findThreadDim(kernelProfiles[CI], *F, dyn_cast<LoadInst>(CI->getArgOperand(2)), true);
            }
            //find host kernel function and actual kernel name
            Module *M = F->getParent();
            auto hostKernel = kernelCall->getCalledFunction();
            funcs2delete.insert(hostKernel);
            StringRef host_kernelName = hostKernel->getName();
            auto namePair = host_kernelName.rsplit("_CudaFE_");
            StringRef kernelName = namePair.second;
            if(kernelName == "") kernelName = host_kernelName;
            errs() << "mergeKernel: found kernelName: " << kernelName << "\n";

            //Find device kernel function
            for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI) {
              Function *F = &*FI;
              auto funcName = F->getName();
              if(funcName.contains(kernelName) &&
                 funcName != host_kernelName ){
                kernelProfiles[CI]->deviceKernel = F;
                deviceKernel = F;
                funcs2delete.insert(deviceKernel);
                break;
              }
            }
            errs() << "MergeKernel: found deviceKernel: " << *deviceKernel << "\n";
            if(device2newFunc.find(deviceKernel) != device2newFunc.end()){
              auto newFunc = device2newFunc[deviceKernel];
              kernelProfiles[CI]->newFunc = newFunc;
              insts2Remove.push_back(CI);
              continue;
            }


            //create a new function for kernel
            Type* i32Ty = Type::getInt32Ty(deviceKernel->getContext());
            Type* i64Ty = Type::getInt64Ty(deviceKernel->getContext());
            std::vector<Type*>argTys;
            for(auto arg = deviceKernel->arg_begin(); arg != deviceKernel->arg_end(); ++arg)
              argTys.push_back(arg->getType());
            for(int i=0; i<12; i++)
              argTys.push_back(i32Ty);
            FunctionType* funcTy = FunctionType::get(
                  deviceKernel->getReturnType(), //return type
                  ArrayRef<Type*>(argTys), //arg types;
                  false
                );
            std::string newName = demangle(deviceKernel->getName());
            // Remove null bytes and other invalid characters from demangled name
            newName.erase(std::remove(newName.begin(), newName.end(), '\0'), newName.end());
            newName = newName.substr(0, newName.find("("));
            kernelProfiles[CI]->newFunc = Function::Create(
                  funcTy,
                  deviceKernel->getLinkage(),
                  newName,
                  //deviceKernel->getName(),
                  deviceKernel->getParent()
                );
            newFunc = kernelProfiles[CI]->newFunc;
            //errs() << "DEVICEKERNEL " << deviceKernel->getName() << "\n";
            //errs() << "NEWFUNC      " << newFunc->getName() << "\n";
            device2newFunc[deviceKernel] = newFunc;
            deviceKernel->setSubprogram(nullptr);

            //copy old kernel over to the new
            ValueToValueMapTy VMap;
            auto NewFArgIt = newFunc->arg_begin();
            for (auto &Arg: deviceKernel->args()) {
              auto ArgName = Arg.getName();
              NewFArgIt->setName(ArgName);
              VMap[&Arg] = &(*NewFArgIt++);
            }

            errs() << "MergeKernel: created new Function: " << *(newFunc) << "\n";
            SmallVector<ReturnInst*, 8> Returns;
            llvm::CloneFunctionInto(newFunc, deviceKernel, VMap, false, Returns);
            errs() << *(newFunc) << "\n";

            // find sync insts inside new kernel
            auto syncInstList = syncInsts[deviceKernel];
            for(auto* inst: syncInstList) {
              if(isa<Instruction>(VMap[inst]))
                syncInsts[newFunc].insert(cast<Instruction>(VMap[inst]));
            }
            syncInsts.erase(deviceKernel);
            
            ////remove the use of the argument in device kernel
            //std::vector<Instruction*> uses2remove;
            //for(auto arg = deviceKernel->arg_begin(); arg != deviceKernel->arg_end(); ++arg) {
            //  auto newArg = VMap[arg];
            //  for(User *U : newArg->users()){
            //    if(Instruction *UI = dyn_cast<Instruction>(U)){
            //      uses2remove.push_back(UI);
            //    }
            //  }
            //}
            //for(auto I : uses2remove)
            //  I->eraseFromParent();

            //set argument name in new kernel
            int i = 0;
            for(auto arg = newFunc->arg_begin(); arg != newFunc->arg_end(); ++arg) {
              bool foundOldArg = false;
              for(auto [oldArg, newArg] : VMap){
                if(newArg == arg){
                  foundOldArg = true;
                  break;
                }
              }
              if(foundOldArg) continue;

              switch(i){
                case 0:
                  arg->setName("gridDim.x");
                  break;
                case 1:
                  arg->setName("gridDim.y");
                  break;
                case 2:
                  arg->setName("gridDim.z");
                  break;
                case 3:
                  arg->setName("blockDim.x");
                  break;
                case 4:
                  arg->setName("blockDim.y");
                  break;
                case 5:
                  arg->setName("blockDim.z");
                  break;
                case 6:
                  arg->setName("blockIdx.x");
                  break;
                case 7:
                  arg->setName("blockIdx.y");
                  break;
                case 8:
                  arg->setName("blockIdx.z");
                  break;
                case 9:
                  arg->setName("threadIdx.x");
                  break;
                case 10:
                  arg->setName("threadIdx.y");
                  break;
                case 11:
                  arg->setName("threadIdx.z");
                  break;
              }
              ++i;
            }

            //replace the use of llvm.nvvm.read.ptx.sreg.* by new arguments
            //TODO: map doesn't accomodate multiple calls
            std::vector<Instruction*> cudaCall2remove;
            for(inst_iterator I = inst_begin(newFunc),
                E = inst_end(newFunc); I != E; ++I){
              CallInst *ci = dyn_cast<CallInst>(&*I);
              if(!ci) continue;
              Function *calledFunc = ci->getCalledFunction();
              if (!calledFunc) continue; // Skip indirect calls and external functions
              auto calledFuncName = calledFunc->getName();

              std::map<std::string, std::string> nvvmCall2arg {
                { "llvm.nvvm.read.ptx.sreg.tid.x", "threadIdx.x"},
                { "llvm.nvvm.read.ptx.sreg.tid.y", "threadIdx.y"},
                { "llvm.nvvm.read.ptx.sreg.tid.z", "threadIdx.z"},
                { "llvm.nvvm.read.ptx.sreg.ntid.x", "blockDim.x"},
                { "llvm.nvvm.read.ptx.sreg.ntid.y", "blockDim.y"},
                { "llvm.nvvm.read.ptx.sreg.ntid.z", "blockDim.z"},
                { "llvm.nvvm.read.ptx.sreg.ctaid.x", "blockIdx.x"},
                { "llvm.nvvm.read.ptx.sreg.ctaid.y", "blockIdx.y"},
                { "llvm.nvvm.read.ptx.sreg.ctaid.z", "blockIdx.z"},
                { "llvm.nvvm.read.ptx.sreg.nctaid.x", "gridDim.x"},
                { "llvm.nvvm.read.ptx.sreg.nctaid.y", "gridDim.y"},
                { "llvm.nvvm.read.ptx.sreg.nctaid.z", "gridDim.z"}
              };

              if(nvvmCall2arg.find(calledFuncName) == nvvmCall2arg.end()) continue;
              auto argName = nvvmCall2arg[calledFuncName];

              cudaCall2remove.push_back(ci);
              for(auto arg = newFunc->arg_begin(); arg != newFunc->arg_end(); ++arg){
                if(arg->getName() != argName) continue;
                for(User *U : ci->users()){
                  Instruction *inst = dyn_cast<Instruction>(U);
                  if(!inst) continue;
                  for (auto OI = inst->op_begin(), OE = inst->op_end(); OI != OE; ++OI){
                    Value *val = *OI;
                    if(val == ci)
                      *OI = arg;
                  }
                }
              }
            }

            for(auto I : cudaCall2remove){
              errs() << "mergeKernel: cudaCall2remove: " << *I << "\n";
              errs() << "mergeKernel: what do I belong to?\n\t" <<  I->getParent()->getName();
              I->eraseFromParent();
            }
            insts2Remove.push_back(CI);
            funcs2delete.insert(calledFunc);
          }
          else if(calledFunc->getName().contains("cudaMalloc")){
            // Lower cudaMalloc to host malloc+store before removing CUDA call.
            bool replacedCudaMalloc = false;
            if (CI->arg_size() >= 2) {
              Value *devDataPtr = CI->getArgOperand(0);
              Value *allocSize = CI->getArgOperand(1);
              if (auto *bitcast = dyn_cast<BitCastInst>(devDataPtr))
                devDataPtr = bitcast->getOperand(0);

              auto *devDataPtrTy = dyn_cast<PointerType>(devDataPtr->getType());
              if (devDataPtrTy && devDataPtrTy->getElementType()->isPointerTy()) {
                Type *targetPtrTy = devDataPtrTy->getElementType();
                Type *i64Ty = Type::getInt64Ty(CI->getContext());
                Value *allocSize64 = allocSize;
                if (allocSize64->getType() != i64Ty) {
                  if (allocSize64->getType()->isIntegerTy())
                    allocSize64 = CastInst::CreateIntegerCast(
                        allocSize64, i64Ty, false, "malloc.size.cast", CI);
                  else
                    allocSize64 = ConstantInt::get(i64Ty, 0);
                }
                FunctionCallee mallocCallee = F->getParent()->getOrInsertFunction(
                    "malloc", Type::getInt8PtrTy(CI->getContext()), i64Ty);
                auto *mallocCall =
                    CallInst::Create(mallocCallee, {allocSize64}, "tulip.host.malloc", CI);
                Value *castedAlloc = mallocCall;
                if (targetPtrTy != mallocCall->getType())
                  castedAlloc = CastInst::CreateBitOrPointerCast(
                      mallocCall, targetPtrTy, "tulip.host.malloc.cast", CI);
                new StoreInst(castedAlloc, devDataPtr, CI);
                replacedCudaMalloc = true;
                errs() << "mergeKernel: replaced cudaMalloc call with host malloc: " << *CI << "\n";
              }
            }

            if (replacedCudaMalloc) {
              funcs2delete.insert(calledFunc);
              if(!CI->use_empty())
                CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0));
              insts2Remove.push_back(CI);
            } else {
              errs() << "mergeKernel: WARN: could not replace cudaMalloc safely, keeping call: "
                     << *CI << "\n";
            }
          }
          else if(calledFunc->getName().contains("cudaMemcpyToSymbol")){
            // Lower cudaMemcpyToSymbol(symbol, src, count[, offset[, kind]])
            // to a plain memcpy into the target global/symbol on host.
            if (CI->arg_size() < 3) {
              errs() << "mergeKernel: WARN: unexpected cudaMemcpyToSymbol signature, skipping transform: "
                     << *CI << "\n";
              continue;
            }

            LLVMContext &C = CI->getContext();
            Type *i8Ty = Type::getInt8Ty(C);
            Type *i64Ty = Type::getInt64Ty(C);

            Value *dstSymbol = CI->getArgOperand(0);
            Value *srcPtr = CI->getArgOperand(1);
            Value *cpySize = CI->getArgOperand(2);
            Value *offset = (CI->arg_size() >= 4)
                                ? CI->getArgOperand(3)
                                : ConstantInt::get(i64Ty, 0);

            // Only support host->device/host->symbol style lowering here.
            // If kind is present and not constant(1), keep conservative behavior.
            if (CI->arg_size() >= 5) {
              if (ConstantInt *kind = dyn_cast<ConstantInt>(CI->getArgOperand(4))) {
                if (!kind->isOne()) {
                  errs() << "mergeKernel: WARN: cudaMemcpyToSymbol kind != cudaMemcpyHostToDevice, skipping: "
                         << *CI << "\n";
                  continue;
                }
              } else {
                errs() << "mergeKernel: WARN: non-constant cudaMemcpyToSymbol kind, skipping: "
                       << *CI << "\n";
                continue;
              }
            }

            auto peelToBaseObject = [](Value *V) -> Value * {
              while (V) {
                if (auto *CE = dyn_cast<ConstantExpr>(V)) {
                  if (CE->isCast() || CE->getOpcode() == Instruction::GetElementPtr) {
                    V = CE->getOperand(0);
                    continue;
                  }
                }
                if (auto *BC = dyn_cast<BitCastInst>(V)) {
                  V = BC->getOperand(0);
                  continue;
                }
                if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
                  V = ASC->getOperand(0);
                  continue;
                }
                if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
                  V = GEP->getPointerOperand();
                  continue;
                }
                break;
              }
              return V;
            };

            auto resolveSymbolShadow = [&](Value *V) -> Value * {
              Value *base = peelToBaseObject(V);
              GlobalVariable *GV = dyn_cast<GlobalVariable>(base);
              if (!GV)
                return V;

              StringRef name = GV->getName();
              size_t dotPos = name.rfind('.');
              if (dotPos == StringRef::npos || dotPos == 0 || dotPos + 1 >= name.size())
                return V;

              StringRef suffix = name.substr(dotPos + 1);
              if (!std::all_of(suffix.begin(), suffix.end(), [](char ch) {
                    return ch >= '0' && ch <= '9';
                  }))
                return V;

              StringRef baseName = name.substr(0, dotPos);
              GlobalVariable *realSymbol = F->getParent()->getNamedGlobal(baseName);
              if (!realSymbol)
                return V;
              if (realSymbol->getValueType() != GV->getValueType())
                return V;

              errs() << "mergeKernel: resolved cudaMemcpyToSymbol shadow global "
                     << GV->getName() << " -> " << realSymbol->getName() << "\n";
              return realSymbol;
            };

            dstSymbol = resolveSymbolShadow(dstSymbol);

            auto *dstPtrTy = dyn_cast<PointerType>(dstSymbol->getType());
            if (!dstPtrTy) {
              errs() << "mergeKernel: WARN: cudaMemcpyToSymbol destination is not a pointer, skipping: "
                     << *CI << "\n";
              continue;
            }
            auto *srcPtrTy = dyn_cast<PointerType>(srcPtr->getType());
            if (!srcPtrTy) {
              errs() << "mergeKernel: WARN: cudaMemcpyToSymbol source is not a pointer, skipping: "
                     << *CI << "\n";
              continue;
            }

            Value *offset64 = offset;
            if (offset64->getType() != i64Ty) {
              if (offset64->getType()->isIntegerTy())
                offset64 = CastInst::CreateIntegerCast(offset64, i64Ty, false, "tulip.sym.off.cast", CI);
              else
                offset64 = ConstantInt::get(i64Ty, 0);
            }

            Value *size64 = cpySize;
            if (size64->getType() != i64Ty) {
              if (size64->getType()->isIntegerTy())
                size64 = CastInst::CreateIntegerCast(size64, i64Ty, false, "tulip.sym.size.cast", CI);
              else
                size64 = ConstantInt::get(i64Ty, 0);
            }

            // Only lower full-symbol writes (offset 0, size == sizeof(symbol)).
            ConstantInt *offImm = dyn_cast<ConstantInt>(offset64);
            ConstantInt *sizeImm = dyn_cast<ConstantInt>(size64);
            if (!offImm || !offImm->isZero() || !sizeImm) {
              errs() << "mergeKernel: WARN: unsupported non-full cudaMemcpyToSymbol, keeping call: "
                     << *CI << "\n";
              continue;
            }
            Type *dstElemTy = dstPtrTy->getPointerElementType();
            const DataLayout &DL = F->getParent()->getDataLayout();
            uint64_t dstElemSize = DL.getTypeStoreSize(dstElemTy);
            if (sizeImm->getZExtValue() != dstElemSize) {
              errs() << "mergeKernel: WARN: cudaMemcpyToSymbol size does not match symbol size, keeping call: "
                     << *CI << "\n";
              continue;
            }
            if (dstElemTy->isAggregateType()) {
              Type *dstBytePtrTy = Type::getInt8PtrTy(C, dstPtrTy->getAddressSpace());
              Type *srcBytePtrTy = Type::getInt8PtrTy(C, srcPtrTy->getAddressSpace());
              Value *dstBytes = dstSymbol;
              Value *srcBytes = srcPtr;
              if (dstBytes->getType() != dstBytePtrTy)
                dstBytes = CastInst::CreatePointerCast(
                    dstBytes, dstBytePtrTy, "tulip.sym.dst.byte.cast", CI);
              if (srcBytes->getType() != srcBytePtrTy)
                srcBytes = CastInst::CreatePointerCast(
                    srcBytes, srcBytePtrTy, "tulip.sym.src.byte.cast", CI);

              uint64_t copyBytes = sizeImm->getZExtValue();
              for (uint64_t b = 0; b < copyBytes; ++b) {
                Value *idx = ConstantInt::get(i64Ty, b);
                Value *dstByte = GetElementPtrInst::Create(
                    i8Ty, dstBytes, {idx}, "tulip.sym.dst.byte", CI);
                Value *srcByte = GetElementPtrInst::Create(
                    i8Ty, srcBytes, {idx}, "tulip.sym.src.byte", CI);
                auto *loadedByte = new LoadInst(i8Ty, srcByte, "tulip.sym.byte.ld", CI);
                new StoreInst(loadedByte, dstByte, CI);
              }
            } else {
              Type *srcTypedPtrTy =
                  PointerType::get(dstElemTy, srcPtrTy->getAddressSpace());
              Value *srcTyped = srcPtr;
              if (srcTyped->getType() != srcTypedPtrTy)
                srcTyped = CastInst::CreatePointerCast(
                    srcTyped, srcTypedPtrTy, "tulip.sym.src.typed.cast", CI);
              auto *typedLoad = new LoadInst(
                  dstElemTy, srcTyped, "tulip.sym.typed.ld", CI);
              new StoreInst(typedLoad, dstSymbol, CI);
            }

            funcs2delete.insert(calledFunc);
            if (!CI->use_empty())
              CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0));
            insts2Remove.push_back(CI);
          }
          else if(calledFunc->getName().contains("cudaMemset")){
            // Lower cudaMemset(dst, value, size) to llvm.memset on host.
            // IR commonly appears as:
            //   call i32 @cudaMemset(i8* (bitcast/load ...), i32 val, i64 size)
            // We must memset the pointed-to allocation, not the pointer slot.
            if (CI->arg_size() < 3) {
              errs() << "mergeKernel: WARN: unexpected cudaMemset signature, skipping transform: "
                     << *CI << "\n";
              continue;
            }

            LLVMContext &C = CI->getContext();
            Type *i8Ty = Type::getInt8Ty(C);
            Type *i8PtrTy = Type::getInt8PtrTy(C);
            Type *i64Ty = Type::getInt64Ty(C);

            Value *dstPtr = CI->getArgOperand(0);
            Value *fillVal = CI->getArgOperand(1);
            Value *setSize = CI->getArgOperand(2);

            // Prefer the loaded pointer if dst is a cast of a load:
            //   bitcast (load T*, T** @slot) to i8*
            if (auto *BC = dyn_cast<BitCastInst>(dstPtr)) {
              if (auto *LD = dyn_cast<LoadInst>(BC->getOperand(0)))
                dstPtr = LD;
              else
                dstPtr = BC->getOperand(0);
            } else if (auto *CE = dyn_cast<ConstantExpr>(dstPtr)) {
              if (CE->isCast() || CE->getOpcode() == Instruction::GetElementPtr)
                dstPtr = CE->getOperand(0);
            }

            if (dstPtr->getType() != i8PtrTy)
              dstPtr = CastInst::CreatePointerCast(dstPtr, i8PtrTy, "tulip.memset.dst.cast", CI);

            Value *fillI8 = fillVal;
            if (fillI8->getType() != i8Ty) {
              if (fillI8->getType()->isIntegerTy())
                fillI8 = CastInst::CreateIntegerCast(fillI8, i8Ty, false, "tulip.memset.val.cast", CI);
              else
                fillI8 = ConstantInt::get(i8Ty, 0);
            }

            Value *size64 = setSize;
            if (size64->getType() != i64Ty) {
              if (size64->getType()->isIntegerTy())
                size64 = CastInst::CreateIntegerCast(size64, i64Ty, false, "tulip.memset.size.cast", CI);
              else
                size64 = ConstantInt::get(i64Ty, 0);
            }
            ConstantInt *sizeImm = dyn_cast<ConstantInt>(size64);
            if (!sizeImm) {
              errs() << "mergeKernel: WARN: non-constant cudaMemset size, keeping call: "
                     << *CI << "\n";
              continue;
            }
            uint64_t setBytes = sizeImm->getZExtValue();
            for (uint64_t b = 0; b < setBytes; ++b) {
              Value *idx = ConstantInt::get(i64Ty, b);
              Value *dstByte = GetElementPtrInst::Create(
                  i8Ty, dstPtr, {idx}, "tulip.memset.byte.ptr", CI);
              new StoreInst(fillI8, dstByte, CI);
            }

            funcs2delete.insert(calledFunc);
            if (!CI->use_empty())
              CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0));
            insts2Remove.push_back(CI);
          }
          else if(calledFunc->getName().contains("cudaMemcpy")){
            if (CI->arg_size() < 4) {
              errs() << "mergeKernel: WARN: unexpected cudaMemcpy signature, skipping transform: "
                     << *CI << "\n";
              continue;
            }
            ConstantInt* mode = dyn_cast<ConstantInt>(CI->getArgOperand(3));
            if (!mode) {
              errs() << "mergeKernel: WARN: non-constant cudaMemcpy mode, skipping transform: "
                     << *CI << "\n";
              continue;
            }
            Value* originalMem = mode->isOne() ? CI->getArgOperand(1) : CI->getArgOperand(0);
            LoadInst* originalLd = nullptr;
            Value* originalAlloc = nullptr;
            if(Instruction* originalBitcast = dyn_cast<BitCastInst>(originalMem)) 
              originalLd = dyn_cast<LoadInst>(originalBitcast->getOperand(0));
            else
              originalLd = dyn_cast<LoadInst>(originalMem);
            
            if(originalLd)
              originalAlloc = originalLd->getOperand(0);
            else
              originalAlloc = originalMem;
              
            auto peelToBaseObject = [](Value *V) -> Value * {
              while (V) {
                if (auto *CE = dyn_cast<ConstantExpr>(V)) {
                  if (CE->isCast() || CE->getOpcode() == Instruction::GetElementPtr) {
                    V = CE->getOperand(0);
                    continue;
                  }
                }

                if (auto *BC = dyn_cast<BitCastInst>(V)) {
                  V = BC->getOperand(0);
                  continue;
                }

                if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
                  V = ASC->getOperand(0);
                  continue;
                }

                if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
                  V = GEP->getPointerOperand();
                  continue;
                }

                break;
              }

              return V;
            };

            originalAlloc = peelToBaseObject(originalAlloc);


            Value* devMem = mode->isOne() ? CI->getArgOperand(0) : CI->getArgOperand(1);
            LoadInst* devLd = nullptr;
            Value* devAlloc = nullptr;
            if(Instruction* devBitcast = dyn_cast<BitCastInst>(devMem)) 
              devLd = dyn_cast<LoadInst>(devBitcast->getOperand(0));
            else
              devLd = dyn_cast<LoadInst>(devMem);
            if(devLd)
              devAlloc = devLd->getOperand(0);
            else
              devAlloc = devMem;

            devAlloc = peelToBaseObject(devAlloc);
            
            bool validOriginal = isa<AllocaInst>(originalAlloc) || isa<GlobalVariable>(originalAlloc);
            bool validDev = isa<AllocaInst>(devAlloc) || isa<GlobalVariable>(devAlloc);

            if(!validOriginal || !validDev){
               errs() << "ANDREW: mergeKernel: invalid memory object for cudaMemcpy replacement logic\n";
               if(originalAlloc) errs() << "Original: " << *originalAlloc << "\n";
               if(devAlloc) errs() << "Dev: " << *devAlloc << "\n";
            }

            if(!validDev || !validOriginal){
              errs() << "ANDREW: mergeKernel: skipping cudaMemcpy rewrite due to unsupported memory objects\n";
              continue;
            }

            Function *memcpyFunc = CI->getFunction();
            auto isSafeMemcpyLoadRewrite = [&](LoadInst *ld) -> bool {
              if (!ld)
                return false;
              Function *ldFunc = ld->getFunction();
              if (ldFunc != memcpyFunc) {
                // Keep alloca-backed rewrites strictly intra-function to avoid
                // leaking stack values across function boundaries. For global-
                // backed mappings, allow cross-function rewrites (CG uses this).
                if (isa<AllocaInst>(originalAlloc) || isa<AllocaInst>(devAlloc)) {
                  errs() << "mergeKernel: skipping cross-function load rewrite in cudaMemcpy: "
                         << (ldFunc ? ldFunc->getName() : "<null>")
                         << " vs " << (memcpyFunc ? memcpyFunc->getName() : "<null>") << "\n";
                  return false;
                }
                errs() << "mergeKernel: allowing cross-function global load rewrite in cudaMemcpy: "
                       << (ldFunc ? ldFunc->getName() : "<null>")
                       << " vs " << (memcpyFunc ? memcpyFunc->getName() : "<null>") << "\n";
              }

              if (AllocaInst *origAlloca = dyn_cast<AllocaInst>(originalAlloc)) {
                if (origAlloca->getFunction() != ldFunc) {
                  errs() << "mergeKernel: skipping unsafe alloca rewrite in cudaMemcpy: alloca in "
                         << origAlloca->getFunction()->getName()
                         << ", load in " << (ldFunc ? ldFunc->getName() : "<null>") << "\n";
                  return false;
                }
              }
              return true;
            };

            std::vector<LoadInst*> devAllocLoadUsers;
            for (User *user : devAlloc->users()) {
              if (LoadInst *ld = dyn_cast<LoadInst>(user)) {
                if (isSafeMemcpyLoadRewrite(ld))
                  devAllocLoadUsers.push_back(ld);
              }
            }
            
            if(originalLd)
              errs() << "ANDREW: mergeKernel: originalLd " << *originalLd << "\n";
            else
              errs() << "ANDREW: mergeKernel: originalLd is NULL, originalMem: " << *originalMem << "\n";
            
            Value* cpySize = nullptr;
            errs() << "mergeKernel: found originalAlloc " << *originalAlloc << "\n";
            for(auto user : originalAlloc->users()){
              if(StoreInst *st = dyn_cast<StoreInst>(user)){
              }
            }
            for(auto user : originalAlloc->users()){
              errs() << "mergeKernel: found originalAlloc user" << *user << "\n";
              if(StoreInst *st = dyn_cast<StoreInst>(user)){
                errs() << "mergeKernel: found originalAlloc store:" << *st << "\n";
                if(BitCastInst* cast = dyn_cast<BitCastInst>(st->getOperand(0))){
                  errs() << "mergeKernel: found originalAlloc cast:" << *cast << "\n";
                  if(CallInst* ci = dyn_cast<CallInst>(cast->getOperand(0))){
                    errs() << "mergeKernel: found originalAlloc ci:" << *ci << "\n";
                    Function *calledCI = ci->getCalledFunction();
                    if(!calledCI || !calledCI->getName().contains("malloc")) continue;
                    cpySize = ci->getArgOperand(0);
                    LLVMContext &C = cpySize->getContext();
                    MDNode *mdSize = nullptr;
                    if(Instruction* sizeInst = dyn_cast<Instruction>(cpySize)){
                      std::string mdDataSize = "tulip.target.datasize";
                      mdSize = MDNode::get(C, MDString::get(C, std::to_string(mdID)));
                      sizeInst->setMetadata("tulip.target.datasize", mdSize);
                      mdID++;
                    }
                    MDNode* N;
                    mdSize ? N = MDNode::get(C, mdSize) :
                             N = MDNode::get(C, ValueAsMetadata::get(dyn_cast<ConstantInt>(cpySize)));
                    mode->isOne() ? ci->setMetadata("tulip.target.mapdata.to", N) :
                                    ci->setMetadata("tulip.target.mapdata.from", N);
                  }
                }
              }
            }
            // For array/pointer-backed host mappings, rewrite loads of the
            // corresponding device pointer:
            //  - static arrays => GEP to first element
            //  - pointer-backed globals/allocas (CG-style) => load from host ptr slot
            // Keep scalar/global mappings (e.g. passed_verification in IS) untouched
            // to avoid castHost pollution.
            Type* originalPointeeType = originalAlloc->getType()->getPointerElementType();
            if (ArrayType *arrTy = dyn_cast<ArrayType>(originalPointeeType)) {
              (void)arrTy;
              for (LoadInst *ld : devAllocLoadUsers) {
                if (ld->use_empty()) {
                  errs() << "mergeKernel: skipping empty load during static array rewrite: " << *ld << "\n";
                  continue;
                }
                Type* i64Ty = Type::getInt64Ty(ld->getContext());
                std::vector<Value*> idxs = {
                  ConstantInt::get(i64Ty, 0),
                  ConstantInt::get(i64Ty, 0)
                };
                GetElementPtrInst* gep = GetElementPtrInst::Create(
                  originalPointeeType,
                  originalAlloc,
                  idxs,
                  "staticArrayPtr",
                  ld
                );
                errs() << "mergeKernel: replacing load with GEP for static array: " << *gep << "\n";
                ld->replaceAllUsesWith(gep);
                ld->eraseFromParent();
              }
            } else if (originalPointeeType->isPointerTy()) {
              for (LoadInst *ld : devAllocLoadUsers) {
                if (ld->use_empty()) {
                  errs() << "mergeKernel: skipping empty load during pointer rewrite: " << *ld << "\n";
                  continue;
                }
                if (ld->getType() == originalPointeeType) {
                  ld->setOperand(0, originalAlloc);
                } else {
                  auto cast = CastInst::CreatePointerCast(
                      originalAlloc, ld->getOperand(0)->getType(), "castHost", ld);
                  ld->setOperand(0, cast);
                }
              }
            } else if (devLd) {
              errs() << "mergeKernel: skipping scalar cudaMemcpy load rewrite: "
                     << *devLd << "\n";
            }




            //TODO: better to create an empty function that doesn't really do anything
            if(mode->isOne()){

              bool foundMapRegion = false;
              for(auto &I : *(CI->getParent())){
                if(I.getMetadata("tulip.target.start.of.map")){
                  foundMapRegion = true;
                  insts2Remove.push_back(CI);
                  break;
                }
              }
              if(foundMapRegion) continue;
              LLVMContext &C = CI->getContext();
              MDNode *N = MDNode::get(C, MDString::get(C,""));
              CI->setMetadata("tulip.target.start.of.map", N);
            }
            else{
              bool foundMapRegion = false;
              for(auto &I : *(CI->getParent())){
                if(I.getMetadata("tulip.target.end.of.map")){
                  foundMapRegion = true;
                  insts2Remove.push_back(CI);
                  break;
                }
              }
              if(foundMapRegion) continue;
              LLVMContext &C = CI->getContext();
              MDNode *N = MDNode::get(C, MDString::get(C,""));
              CI->setMetadata("tulip.target.end.of.map", N);
            }
            //errs() << "mergeKernel: cudaMemcpy: " << *CI << "\n";
            //Type* Int1Ty = Type::getInt1Ty(F->getContext());
            //CallSite CS(CI);
            //SmallVector<Value *, 4> Args(CS.arg_begin(), CS.arg_end()-1);
            //Args.push_back(ConstantInt::getFalse(Int1Ty));
            //ArrayRef<Value*> args(Args);
            //std::vector<Type*> argTyVec;
            //argTyVec.push_back(PointerType::get(Type::getInt8Ty(F->getContext()), 0));
            //argTyVec.push_back(PointerType::get(Type::getInt8Ty(F->getContext()), 0));
            //argTyVec.push_back(Type::getInt64Ty(F->getContext()));
            //argTyVec.push_back(Int1Ty);
            //ArrayRef<Type *> argTys(argTyVec);
            //FunctionType* memcpyFuncTy = FunctionType::get(
            //    Type::getVoidTy(F->getContext()), //return type
            //    argTys,
            //    false
            //);

            //auto MemCpyFunc = F->getParent()->getOrInsertFunction("llvm.memcpy.p0i8.p0i8.i64", memcpyFuncTy);
            //CallInst *NewCI = CallInst::Create(MemCpyFunc, args, "", CI);

            ////add metadata to identify omp target mapping
            ////Instruction* src = CI->getArgOperand(1);
            //Value* cpySize = CI->getArgOperand(2);
            //ConstantInt* mode = dyn_cast<ConstantInt>(CI->getArgOperand(3));
            //Instruction* dest = nullptr;
            //mode->isOne()? dest = dyn_cast<Instruction>(CI->getArgOperand(0)) :
            //               dest = dyn_cast<Instruction>(CI->getArgOperand(1));
            //LLVMContext& C = NewCI->getContext();
            //std::string mdDevice = "tulip.target.mapdata";
            //MDNode *mdSize = nullptr;
            //if(Instruction* sizeInst = dyn_cast<Instruction>(cpySize)){
            //  std::string mdDataSize = "tulip.target.datasize";
            //  mdSize = MDNode::get(C, MDString::get(C, std::to_string(mdID)));
            //  sizeInst->setMetadata("tulip.target.datasize", mdSize);
            //  mdID++;
            //}

            //MDNode* N;
            //mdSize ? N = MDNode::get(C, mdSize) :
            //         N = MDNode::get(C, ValueAsMetadata::get(dyn_cast<ConstantInt>(cpySize)));
            //mode->isOne() ? dest->setMetadata("tulip.target.mapdata.to", N) :
            //                dest->setMetadata("tulip.target.mapdata.from", N);
          }
          //TODO: handle cudaDeviceSynchronize
          else if(calledFunc->getName().contains("cudaDeviceSynchronize")){
            insts2Remove.push_back(CI);
          }
          else if (calledFunc->getName().contains("cudaSetDevice")){
            insts2Remove.push_back(CI);
          }
          else if (calledFunc->getName().contains("cudaGetDeviceProperties")){
            insts2Remove.push_back(CI);
          }
          else if (calledFunc->getName().contains("cudaGetDeviceCount")){
            insts2Remove.push_back(CI);
          }
          else if (calledFunc->getName().contains("cudaGetDevice")){
            insts2Remove.push_back(CI);
          }
        }
      }


      //create loops around kernel
      for(auto [configureCI, kernelProfile] : kernelProfiles){
        std::vector<Value*> indvars;
        auto loopDims = kernelProfile->loopDims;
        if(kernelProfile->gridLoopCnt + kernelProfile->blockLoopCnt != 0){
          std::stack<BasicBlock*> headerNests, headerNests2;
          auto kernelBB = kernelProfile->kernelBB;
          BasicBlock* pred = kernelBB->getSinglePredecessor();
          std::vector<BasicBlock*> succs(succ_begin(kernelBB), succ_end(kernelBB));
          auto term = dyn_cast<BranchInst>(pred->getTerminator());
          assert(term && "mergeKernel: term is not a branch inst 206\n");
          auto kernelPred = kernelBB->getSinglePredecessor();
          assert(kernelPred && "mergeKernel: kernel has multiple predecessors\n");
          insts2Remove.push_back(term);
          Value *cond = nullptr;
          if(term->isConditional())
            cond = term->getCondition();
          int loopCnt = 0;
          BasicBlock *lastheader = nullptr;

          //create headers
          std::map<BasicBlock*, Value*>header2itNum;
          for(auto itNum : loopDims){
            if(ConstantInt *constInt = dyn_cast<ConstantInt>(itNum))
              if(constInt->getSExtValue() == 1)
                continue;
            auto header = BasicBlock::Create(kernelBB->getContext(), "header." + std::to_string(loopCnt), F, kernelBB);
            headerNests.push(header);
            header2itNum[header] = itNum;
            if(loopCnt == 0){
              auto br = BranchInst::Create(header, pred);
            //  LLVMContext& C = term->getContext();
            //  MDNode* N = MDNode::get(C, MDString::get(C, ""));
            //  term->setMetadata("splendid.doall.loop", N);
            }
            pred = header;
            loopCnt++;
            lastheader = header;
          }
          headerNests2 = headerNests;

          //create latches
          BasicBlock* lastLatch = nullptr;
          std::map<BasicBlock*, BasicBlock*> header2latch;
          for(int i=loopCnt-1; i>=0; --i){
            auto header = headerNests.top();
            auto latch = BasicBlock::Create(kernelBB->getContext(), "latch." + std::to_string(i), F, kernelBB);
            if(!lastLatch)
              lastLatch = latch;
            auto br = BranchInst::Create(header, latch);

            header2latch[header] = latch;
            headerNests.pop();
          }


          //create header branches
          auto loopExit = kernelBB->getSingleSuccessor();
          assert(loopExit && "kernel BB has multiple exits\n");

          BasicBlock *prevHeader = nullptr;
          int i = loopCnt-1;
          while(!headerNests2.empty()){
            auto header = headerNests2.top();
            headerNests2.pop();
            auto nextHeader = headerNests2.empty()? nullptr : headerNests2.top();

            //create phi node
            Type *phiTy = header2itNum[header]->getType();
            auto indvar = PHINode::Create(phiTy, 2, "indvar."+std::to_string(i), header);
            indvars.push_back(indvar);
            CmpInst *cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULT, indvar, header2itNum[header], "exitCheck." + std::to_string(i), header);
            BranchInst *term = nullptr;

            //create increment in latch
            auto latch = header2latch[header];
            Value *incr = BinaryOperator::Create(Instruction::Add, indvar, ConstantInt::get(phiTy, 1),
                                      "indvar.next." + std::to_string(i), latch->getTerminator());

            //TODO: figure out why i need getName empty

            //a single loop
            if(!nextHeader && !prevHeader){
              term = BranchInst::Create(kernelBB, loopExit, cmp, header);
              indvar->addIncoming(ConstantInt::get(phiTy,0), kernelPred);
            }
            else if(!nextHeader || !nextHeader->hasName() || nextHeader->getName() == ""){
              term = BranchInst::Create(prevHeader, loopExit, cmp, header);
              indvar->addIncoming(ConstantInt::get(phiTy,0), kernelPred);
            }
            else if(!prevHeader){
              term = BranchInst::Create(kernelBB, header2latch[nextHeader], cmp, header);
              indvar->addIncoming(ConstantInt::get(phiTy,0), nextHeader);
            }
            else{
              term = BranchInst::Create(prevHeader, header2latch[nextHeader], cmp, header);
              indvar->addIncoming(ConstantInt::get(phiTy,0), nextHeader);
            }

            bool mayFissionKernel = false;
            if (kernelProfile->newFunc &&
                syncInsts.find(kernelProfile->newFunc) != syncInsts.end() &&
                !syncInsts[kernelProfile->newFunc].empty()) {
              mayFissionKernel = true;
            }
            if(kernelProfile->dim2classify[header2itNum[header]] == 1 ||
                kernelProfile->dim2classify[header2itNum[header]] == 2){
              LLVMContext& C = term->getContext();
              MDNode* N = MDNode::get(C, MDString::get(C, ""));
              if(kernelProfile->dim2classify[header2itNum[header]] == 1){
                // Always keep grid-level parallelism so we still emit a pragma.
                // Collapse happens only when both grid and block are tagged.
                term->setMetadata("tulip.doall.loop.grid", N);
              }
              if(kernelProfile->dim2classify[header2itNum[header]] == 2){
                // If kernel is likely fissioned, avoid block tag so codegen
                // cannot form collapse(2) on a non-perfect nest.
                if (!mayFissionKernel)
                  term->setMetadata("tulip.doall.loop.block", N);
                else
                  errs() << "ANDREW: skip block doall metadata for fission-prone kernel to avoid invalid collapse\n";
              }
              errs() << "mergeKernel: create metadata" << *term << "\n";
            }
            indvar->addIncoming(incr, header2latch[header]);
            prevHeader = header;
            i--;
          }

          //replace kernel branch
          insts2Remove.push_back(kernelBB->getTerminator());
          BranchInst::Create(lastLatch, kernelBB);
        }


        std::vector<Value*> newKernelArgs;
        auto kernelCall = kernelProfile->kernelCall;
        errs() << *kernelCall << "\n";
        for(int i=0; i<kernelCall->arg_size(); ++i){
          errs() << "SUSAN: original arg " << *(kernelCall->getArgOperand(i)) << "\n";
          newKernelArgs.push_back(kernelCall->getArgOperand(i));
        }
        std::reverse(indvars.begin(), indvars.end());
        std::vector<Value*> indvarsExtended;
        int i=0;
        auto i32Ty = Type::getInt32Ty(kernelCall->getContext());
        if(loopDims.size() == 2){
          //push dims
          auto dim = loopDims[0];
          auto arg = dim;
          if(!dim->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(dim, i32Ty, "dim.cast", kernelCall);
          newKernelArgs.push_back(arg);
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));
          dim = loopDims[1];
          arg = dim;
          if(!dim->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(dim, i32Ty, "dim.cast", kernelCall);
          newKernelArgs.push_back(arg);
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));
          newKernelArgs.push_back(ConstantInt::get(i32Ty, 1));


          //create indvars
          auto indvar = indvars[0];
          arg = indvar;
          if(!arg->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(indvar, i32Ty, "dim.cast", kernelCall);
          indvarsExtended.push_back(arg);
          indvarsExtended.push_back(ConstantInt::get(i32Ty, 0));
          indvarsExtended.push_back(ConstantInt::get(i32Ty, 0));
          indvar = indvars[1];
          arg = indvar;
          if(!arg->getType()->isIntegerTy(32))
            arg = CastInst::CreateTruncOrBitCast(indvar, i32Ty, "dim.cast", kernelCall);
          indvarsExtended.push_back(arg);
          indvarsExtended.push_back(ConstantInt::get(Type::getInt32Ty(kernelCall->getContext()), 0));
          indvarsExtended.push_back(ConstantInt::get(Type::getInt32Ty(kernelCall->getContext()), 0));
        } else {
          for(auto itNum : loopDims){
            errs() << "SUSAN: itnum: " << *itNum << "\n";
            newKernelArgs.push_back(itNum);
            if(ConstantInt *constInt = dyn_cast<ConstantInt>(itNum))
              if(constInt->getSExtValue() == 1){
                indvarsExtended.push_back(ConstantInt::get(Type::getInt32Ty(kernelCall->getContext()), 0));
                continue;
              }
            indvarsExtended.push_back(indvars[i]);
            ++i;
          }
        }

        for(auto indvar : indvarsExtended){
          errs() << "mergeKernel: indvar " << *indvar << "\n";
          newKernelArgs.push_back(indvar);
        }
        auto newFunc = kernelProfile->newFunc;

        unsigned count = 0;
        for(auto arg : newKernelArgs){
          errs() << "mergeKernel: new function kernel Args: " << *arg << "\n";
          count++;
        }
        errs() << count << "\n";
        
        errs() << newFunc->getFunctionType()->getNumParams() << "\n";
        errs() << newFunc->getName() << " type: " << *(newFunc->getFunctionType()) << "\n";
        for(unsigned i = 0; i < newFunc->getFunctionType()->getNumParams(); i++) {
          errs() << *newFunc->getFunctionType()->getParamType(i) << " " << *newKernelArgs[i] << "\n";
        }

        CallInst *newKernelCall = CallInst::Create(
              newFunc->getFunctionType(), //function type
              newFunc, //function
              ArrayRef<Value*>(newKernelArgs), //args
              "",
              kernelCall //insert before
            );
          insts2Remove.push_back(kernelCall);
          kernelCalls[newFunc].insert(newKernelCall);
          errs() << "Finished making call for " << newFunc->getName() << "\n";
      }

       //delete cuda calls and control flows
       for(auto I : insts2Remove)
         I->eraseFromParent();

    }

    //delete functions
    for(auto f : funcs2delete) {
      f->eraseFromParent();
    }

    // Remove CUDA-side atomic helper implementations from the module.
    // Atomic call-sites are lowered in C backend, so helper bodies should
    // not appear in emitted C.
    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
      Function *F = &*FI;
      if (F->isDeclaration())
        continue;
      if (!F->getName().contains("atomicAdd") &&
          !F->getName().contains("atomicCAS"))
        continue;
      atomicImplFuncs.insert(F);
    }
    for (auto *F : atomicImplFuncs) {
      if (!F || F->isDeclaration())
        continue;
      F->deleteBody();
      F->setLinkage(GlobalValue::ExternalLinkage);
    }


    //process shared variables - lift to thread-local globals in address space 0
     std::map<GlobalVariable*, GlobalVariable*> oldGlob2NewGlob; // Map from addrspace(3) to addrspace(0)
     std::map<ConstantExpr*, GlobalVariable*> addrspacecast2NewGlob; // Map from addrspacecast to new global
     std::map<GEPOperator*, std::vector<Instruction*>> gepOp2Users; // Map GEPOperator to instructions using it
     std::vector<GlobalVariable*> globalsToDelete; // Original address space 3 globals to delete
     std::map<std::string, int> nameCounts; // Track name usage to avoid conflicts
     
    auto extractVarName = [](const std::string& mangledName) -> std::string {
        // Prefer Itanium demangling for globals like _ZL11grid_points.
        std::string demangled = demangle(mangledName);
        if (!demangled.empty() && demangled != mangledName) {
            // Trim any function/signature tail if present.
            size_t parenPos = demangled.find('(');
            if (parenPos != std::string::npos)
                demangled = demangled.substr(0, parenPos);

            // Keep the innermost identifier for namespace/class-qualified names.
            size_t lastSep = demangled.rfind("::");
            if (lastSep != std::string::npos && lastSep + 2 < demangled.length())
                demangled = demangled.substr(lastSep + 2);

            if (!demangled.empty())
                return demangled;
        }

        // Legacy heuristic for partially mangled forms.
        size_t ePos = mangledName.find_last_of('E');
        if (ePos != std::string::npos && ePos + 1 < mangledName.length()) {
            size_t startPos = ePos + 1;
            while (startPos < mangledName.length() &&
                   std::isdigit(static_cast<unsigned char>(mangledName[startPos]))) {
                startPos++;
            }
            if (startPos < mangledName.length())
                return mangledName.substr(startPos);
        }

        return mangledName;
    };
     
     // Find all address space 3 globals and create corresponding address space 0 globals
     for (Module::global_iterator I = M.global_begin(), E = M.global_end();
           I != E; ++I) {
         GlobalVariable* globVal = &*I;
          // Check for address space 3 (shared memory) - handle both initialized and external globals
          // External shared memory (extern __shared__) won't have an initializer but still needs processing
          if(globVal->getAddressSpace() != 3) continue;
         
         PointerType* ty = dyn_cast<PointerType>(globVal->getType());
         if(!ty){
           errs() << "WARNINGS: shared object ty is not a pointer type\n";
           continue;
         }
         
         Type* pointedTy = ty->getPointerElementType();
          
          // Handle extern shared memory which is declared as [0 x T]
          // We need to create a properly sized array for CPU execution
          if(ArrayType* arrTy = dyn_cast<ArrayType>(pointedTy)) {
              if(arrTy->getNumElements() == 0) {
                  // Create a reasonably sized array (1024 elements for max threads per block)
                  Type* elemTy = arrTy->getElementType();
                  pointedTy = ArrayType::get(elemTy, 1024);
                  errs() << "mergeKernel: Replacing zero-length array with [1024 x " << *elemTy << "]\n";
              }
          }
         std::string varName = extractVarName(globVal->getName().str());
         std::string baseName = varName + "_shared";
        std::string newName = baseName;
         int counter = 0;
         while(M.getNamedValue(newName) != nullptr || nameCounts.find(newName) != nameCounts.end()){
           counter++;
           newName = baseName + std::to_string(counter);
         }
         nameCounts[newName] = 1;
         GlobalVariable* newGlob = new GlobalVariable(
             M,
             pointedTy,
             false, // isConstant
             GlobalValue::InternalLinkage,
             Constant::getNullValue(pointedTy), // initializer
             newName,
             nullptr, // insert before
            GlobalVariable::GeneralDynamicTLSModel,
             0 // address space 0
         );
         
        oldGlob2NewGlob[globVal] = newGlob;
        globalsToDelete.push_back(globVal);
        errs() << "mergeKernel: Created new global " << newName << " for shared memory " << globVal->getName() << "\n";
        errs() << "ANDREW: shared-lowering created TLS global " << newGlob->getName()
               << " (old=" << globVal->getName() << ", addrspace3->0)\n";
     }
     
     // Replace all uses of address space 3 globals
     for(auto [oldGlob, newGlob] : oldGlob2NewGlob){
         std::vector<User*> usersToProcess;
         for(User *U : oldGlob->users()){
             usersToProcess.push_back(U);
         }
         
       for(User *U : usersToProcess){
           ConstantExpr *UE = dyn_cast<ConstantExpr>(U);
           if(!UE) continue;

           // Direct oldGlob -> addrspacecast use.
           if(UE->getOpcode() == Instruction::AddrSpaceCast){
               addrspacecast2NewGlob[UE] = newGlob;
               errs() << "mergeKernel: Found addrspacecast: " << *UE << "\n";
               std::vector<User*> castUsers;
               for(User *CastU : UE->users()){
                   castUsers.push_back(CastU);
               }

               for(User *CastU : castUsers){
                   if(GEPOperator *gepOp = dyn_cast<GEPOperator>(CastU)){
                       errs() << "mergeKernel: Found GEPOperator: " << *gepOp << "\n";
                       std::vector<Instruction*> gepUsers;
                       for(User *GepU : gepOp->users()){
                           if(Instruction *I = dyn_cast<Instruction>(GepU)){
                               gepUsers.push_back(I);
                           }
                       }
                       gepOp2Users[gepOp] = gepUsers;
                   }
               }
               continue;
           }

           // Common nested chain: oldGlob -> bitcast -> addrspacecast.
           for(User *NestedU : UE->users()){
               ConstantExpr *NestedCE = dyn_cast<ConstantExpr>(NestedU);
               if(!NestedCE || NestedCE->getOpcode() != Instruction::AddrSpaceCast)
                   continue;
               errs() << "ANDREW: shared-lowering discovered nested addrspacecast via "
                      << *UE << "\n";
               addrspacecast2NewGlob[NestedCE] = newGlob;
               errs() << "mergeKernel: Found addrspacecast: " << *NestedCE << "\n";
               std::vector<User*> castUsers;
               for(User *CastU : NestedCE->users()){
                   castUsers.push_back(CastU);
               }

               for(User *CastU : castUsers){
                   if(GEPOperator *gepOp = dyn_cast<GEPOperator>(CastU)){
                       errs() << "mergeKernel: Found GEPOperator: " << *gepOp << "\n";
                       std::vector<Instruction*> gepUsers;
                       for(User *GepU : gepOp->users()){
                           if(Instruction *I = dyn_cast<Instruction>(GepU)){
                               gepUsers.push_back(I);
                           }
                       }
                       gepOp2Users[gepOp] = gepUsers;
                   }
               }
           }
       }
     }
     
     // Replace GEPOperator uses with new GEP instructions using new globals
     for(auto [gepOp, users] : gepOp2Users){
         if(users.empty()) continue;
         
         GlobalVariable* newGlob = nullptr;
         if(ConstantExpr *castExpr = dyn_cast<ConstantExpr>(gepOp->getPointerOperand())){
             if(addrspacecast2NewGlob.find(castExpr) != addrspacecast2NewGlob.end()){
                 newGlob = addrspacecast2NewGlob[castExpr];
             }
         }
         
         if(!newGlob) continue;
         
         for(Instruction *user : users){
             std::vector<Value*> idxList(gepOp->idx_begin(), gepOp->idx_end());
            // Rebuild the GEP from a base pointer with the same type as the
            // original pointer operand. This keeps index semantics intact for
            // both direct shared arrays ([0 x T]*) and bitcasted struct views.
            Value *gepBase = newGlob;
            Type *origPtrTy = gepOp->getPointerOperandType();
            if (gepBase->getType() != origPtrTy) {
                gepBase = ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                    newGlob, cast<PointerType>(origPtrTy));
            }
            Type* gepElemType = gepOp->getSourceElementType();
             GetElementPtrInst *newGep = GetElementPtrInst::Create(
                 gepElemType,
                gepBase,
                 makeArrayRef(idxList),
                 "sharedMem.gep",
                 user
             );
             
            user->replaceUsesOfWith(gepOp, newGep);
            errs() << "mergeKernel: Replaced GEPOperator use in " << *user << "\n";
            errs() << "ANDREW: shared-lowering rewired GEP user to " << newGlob->getName() << "\n";
         }
     }
     
    // Replace remaining addrspacecast uses with new globals
     for(auto [castExpr, newGlob] : addrspacecast2NewGlob){
        Constant *castReplacement =
            ConstantExpr::getPointerBitCastOrAddrSpaceCast(newGlob, castExpr->getType());
        if(castReplacement != castExpr){
            errs() << "ANDREW: shared-lowering replacing addrspacecast constant globally: "
                   << *castExpr << " -> " << *castReplacement << "\n";
            castExpr->replaceAllUsesWith(castReplacement);
        }

         std::vector<User*> castUsers;
         for(User *CastU : castExpr->users()){
             castUsers.push_back(CastU);
         }
         
         for(User *CastU : castUsers){
             if(Instruction *I = dyn_cast<Instruction>(CastU)){
               I->replaceUsesOfWith(castExpr, castReplacement);
                errs() << "mergeKernel: Replaced addrspacecast use in " << *I << "\n";
               errs() << "ANDREW: shared-lowering rewired addrspacecast to replacement constant\n";
             } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(CastU)) {
                errs() << "ANDREW: shared-lowering unresolved constexpr user of addrspacecast "
                       << *CE << "\n";
                for (User *CEU : CE->users()) {
                  if (Instruction *InstUser = dyn_cast<Instruction>(CEU)) {
                    errs() << "ANDREW:   constexpr feeds instruction " << *InstUser << "\n";
                  } else if (ConstantExpr *NestedCE = dyn_cast<ConstantExpr>(CEU)) {
                    errs() << "ANDREW:   constexpr feeds nested constexpr " << *NestedCE << "\n";
                  } else {
                    errs() << "ANDREW:   constexpr feeds non-inst/non-constexpr user\n";
                  }
                }
             }
         }
     }
     
     // Clean up - remove original address space 3 globals
     for(GlobalVariable* oldGlob : globalsToDelete){
         if(oldGlob->use_empty()){
            errs() << "mergeKernel: Removing old shared memory global " << oldGlob->getName() << "\n";
            errs() << "ANDREW: shared-lowering removed old addrspace(3) global " << oldGlob->getName() << "\n";
             oldGlob->eraseFromParent();
         } else {
             errs() << "mergeKernel: WARNING: Old shared memory global " << oldGlob->getName() << " still has uses\n";
            errs() << "ANDREW: shared-lowering old global still live " << oldGlob->getName()
                   << ", remaining uses=" << oldGlob->getNumUses() << "\n";
            for (User *U : oldGlob->users()) {
              if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
                errs() << "ANDREW: remaining old shared global constexpr use: " << *CE << "\n";
                for (User *CEU : CE->users()) {
                  if (Instruction *InstUser = dyn_cast<Instruction>(CEU)) {
                    errs() << "ANDREW:   used by instruction: " << *InstUser << "\n";
                  } else if (ConstantExpr *NestedCE = dyn_cast<ConstantExpr>(CEU)) {
                    errs() << "ANDREW:   used by nested constexpr: " << *NestedCE << "\n";
                  } else {
                    errs() << "ANDREW:   used by unknown user kind\n";
                  }
                }
              } else if (Instruction *I = dyn_cast<Instruction>(U)) {
                errs() << "ANDREW: remaining old shared global instruction use: " << *I << "\n";
              } else {
                errs() << "ANDREW: remaining old shared global unknown user kind\n";
              }
            }
         }
     }

    //Split function at synchronization points
    for(auto [func, insts]: syncInsts) {
      splitFunction(M.getContext(), func, insts);
      for(auto callinst: kernelCalls[func])
        splitLoop(M.getContext(), callinst, insts.size());
    }
    
    return true;
  }
}; // end of struct Hello
}  // end of anonymous namespace

char MergeKernel::ID = 0;
static RegisterPass<MergeKernel> X("merge-kernel", "merge cuda kernel back to main file",
                             false /* Only looks at CFG */,
                             false /* Analysis Pass */);
