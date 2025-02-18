//===- OutlineLoop.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "OutlineLoop World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "LoopUtils.h"
#include "llvm/IR/IRBuilder.h"
#include <string>
#include <unordered_set>
using namespace llvm;

#define DEBUG_TYPE "outline-loop"

STATISTIC(OutlineCounter, "Counts number of loops outlined");

namespace {
  typedef std::vector<Function*> FunctionList;
  typedef std::unordered_set<Function*> FunctionSet;

  // OutlineLoop - The first implementation, without getAnalysisUsage.
  struct OutlineLoop : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    OutlineLoop() : ModulePass(ID) {}

    FunctionList outlinedFunctionList;
    FunctionSet outlinedFunctionCallerList;

    bool runOnModule(Module &M) override {

      for(auto &F: M) {
        if(F.isDeclaration()) continue;
        // if the function name has cudakernel it has already been outlined
        if(F.getName().contains("cudakernel")) continue;

        unsigned OutlinedCounterFunction = 0;

        LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
        DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();

        for(auto *L: LI) {
          outlineToKernel(L, DT, &OutlinedCounterFunction);
        }
      }

      for(auto *F: outlinedFunctionList) {
        setNewBounds(F);
      }

      for(auto *F: outlinedFunctionCallerList) {
          auto &C = F->getContext();
          MDNode* MD = MDNode::get(C, MDString::get(C, ""));
          F->setMetadata("tulip.cuda.kernel.caller", MD);
      }

      // TODO: mark kernel region
      // Possibly in CBE
      //for(auto *F: outlinedFunctionCallerList) {
      //  addThreadBlockVars(F);
      //}

      return false;
    }

    //TODO: add grid & block and delete outermost for loop
    //TODO: add if condition
    void outlineToKernel(Loop* L, DominatorTree& DT, unsigned* OutlinedCounterFunction) {
      auto parentFunc = L->getHeader()->getParent();
      if(L->getHeader()->getTerminator()->getMetadata("noelle.doall.loop")) {
        std::string Suffix = "cudakernel"+std::to_string(*OutlinedCounterFunction);
        CodeExtractor Extractor(DT, *L, false, nullptr, nullptr,
            nullptr, Suffix);
        if(auto* outlinedFunction = Extractor.extractCodeRegion()) {
          auto &C = outlinedFunction->getContext();
          MDNode* MD = MDNode::get(C, MDString::get(C, "begin"));
          outlinedFunction->setMetadata("tulip.cuda.kernel", MD);
          //LI.erase(L);
          ++OutlineCounter;
          ++*OutlinedCounterFunction;
        
          outlinedFunctionList.push_back(outlinedFunction);
          outlinedFunctionCallerList.insert(parentFunc);
        }
      }
      else {
        errs() << "\n";
        SmallVector<Loop*, 4> innerLoops;
        for(auto *innerL: *L) {
          outlineToKernel(innerL, DT, OutlinedCounterFunction);
        }
      }
    }

    void setNewBounds(Function* F) {
      ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>(*F).getSE();
      LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();

      for(auto *L: LI) {
        if(L->getHeader()->getTerminator()->getMetadata("noelle.doall.loop")) {
          Value* lb;
          Value* ub;
          auto IV = looputil::getInductionVariable(L, &SE);
          if(LI.getLoopFor(IV->getIncomingBlock(0)) != L)
            lb = IV->getIncomingValue(0);
          else if(LI.getLoopFor(IV->getIncomingBlock(1)) != L)
            lb = IV->getIncomingValue(1);
          auto *condInst = looputil::findCondInst(L);
          ub = condInst->getOperand(1);

          auto *header = L->getHeader();
          if(header == condInst->getParent()) {
            errs() << "\tCOND IS HEADER\n";
            auto *latch = L->getLoopLatch()->getTerminator();
            assert(latch->getNumSuccessors() == 1 && 
              "need exactly one successor!\n");
            auto &phi = cast<PHINode>(*header->phis().begin());
            phi.removeIncomingValue(L->getLoopLatch());
            MDNode* MD = MDNode::get(phi.getContext(), MDString::get(phi.getContext(), ""));
            phi.setMetadata("tulip.cuda.indvar", MD);

            latch->setSuccessor(0, L->getExitBlock());
          }
          else
            errs() << "\tCOND AND HEADER DO NOT MATCH\n";
        }
      }
    }

    void addThreadBlockVars(Function* F) {
      //Add threadsPerBlock at top
      IRBuilder<> builder(cast<Instruction>(F->getEntryBlock().getFirstInsertionPt()));
      auto *allocaInst = builder.CreateAlloca(builder.getInt32Ty(), nullptr, "threadsPerBlock");
      builder.CreateStore(builder.getInt32(256), allocaInst);

      for(auto &BB: *F) {
        for(auto &I: BB) {
          if(auto CallI = dyn_cast<CallInst>(&I))
            if(CallI->getCalledFunction()->getName().contains("cudakernel")) {
              builder.SetInsertPoint(CallI);
              auto *blockAlloca = builder.CreateAlloca(builder.getInt32Ty(), nullptr, "block_x");
              builder.CreateStore(builder.getInt32(32), blockAlloca);
            }
        }
      }
      
    }


    // We don't modify the program, so we preserve all analyses.
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
    }
  };
}

char OutlineLoop::ID = 0;
static RegisterPass<OutlineLoop> X("outline-loop", "OutlineLoop Pass", false, false);
