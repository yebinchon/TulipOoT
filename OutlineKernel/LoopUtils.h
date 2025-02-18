#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
using namespace llvm;

namespace looputil {
PHINode* getInductionVariable(Loop *L, ScalarEvolution *SE) {
  //errs() << "trying to get IV for Loop:" << *L << "\n";
  PHINode *InnerIndexVar = L->getCanonicalInductionVariable();
  if (InnerIndexVar){
    return InnerIndexVar;
  }
  if (L->getLoopLatch() == nullptr || L->getLoopPredecessor() == nullptr){
    errs() << "SUSAN: didn't find IV 788\n";
    return nullptr;
  }
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    PHINode *PhiVar = cast<PHINode>(I);
    errs() << "SUSAN: phi: " << *PhiVar << "\n";
    Type *PhiTy = PhiVar->getType();
    if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
        !PhiTy->isPointerTy()){
      errs() << "SUSAN: didn't find IV 796\n";
      return nullptr;
    }

    const SCEVAddRecExpr *AddRec = nullptr;
    if(SE->isSCEVable(PhiVar->getType()))
        AddRec = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(PhiVar));
    if (!AddRec || !AddRec->isAffine()){
      errs() << "SUSAN: can't find addRec\n";
      continue;
    }
    //const SCEV *Step = AddRec->getStepRecurrence(*SE);
    //if (!isa<SCEVConstant>(Step) || !isa<SCEVSequentialMinMaxExpr>(Step)){
    //  errs() << "SUSAN: step isn't constant\n";
    //  continue;
    //}

    // Found the induction variable.
    // FIXME: Handle loops with more than one induction variable. Note that,
    // currently, legality makes sure we have only one induction variable.
    return PhiVar;
  }
  errs() << "SUSAN: didn't find IV 812\n";
  return nullptr;
}

Instruction* findCondInst(Loop *L){
  auto header = L->getHeader();
  Instruction* term = header->getTerminator();
  errs() << "term 6818: " << *term << "\n";
  BranchInst* brInst = dyn_cast<BranchInst>(term);
  Value *cond = brInst->getCondition();
  if(isa<CmpInst>(cond) || isa<UnaryInstruction>(cond) || isa<BinaryOperator>(cond) || isa<CallInst>(cond)){
    BasicBlock *succ0 = brInst->getSuccessor(0);
    return cast<Instruction>(cond);
  }

  return nullptr;
}

}