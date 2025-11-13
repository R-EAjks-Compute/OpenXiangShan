// Copyright (c) 2024-2025 Beijing Institute of Open Source Chip (BOSC)
// Copyright (c) 2020-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2020-2021 Peng Cheng Laboratory
//
// XiangShan is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          https://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
package xiangshan.frontend.bpu.utage1

import chisel3._
import chisel3.util._
import freechips.rocketchip.util.SeqToAugmentedSeq
import org.chipsalliance.cde.config.Parameters
import scala.math.min
import utility.XSPerfAccumulate
import xiangshan.frontend.PrunedAddr
import xiangshan.frontend.bpu.BasePredictor
import xiangshan.frontend.bpu.BasePredictorIO
import xiangshan.frontend.bpu.BpuTrain
import xiangshan.frontend.bpu.CompareMatrix
import xiangshan.frontend.bpu.FoldedHistoryInfo
import xiangshan.frontend.bpu.HasFastTrainIO
import xiangshan.frontend.bpu.SaturateCounter
import xiangshan.frontend.bpu.history.phr.PhrAllFoldedHistories

/**
 * This module is the implementation of the TAGE (TAgged GEometric history length predictor).
 */
class MicroTage1(implicit p: Parameters) extends BasePredictor with HasMicroTage1Parameters with Helpers {
  class MicroTageIO(implicit p: Parameters) extends BasePredictorIO {
    val foldedPathHist:         PhrAllFoldedHistories = Input(new PhrAllFoldedHistories(AllFoldedHistoryInfo))
    val foldedPathHistForTrain: PhrAllFoldedHistories = Input(new PhrAllFoldedHistories(AllFoldedHistoryInfo))
    val prediction:             MicroTage1Prediction   = Output(new MicroTage1Prediction)
  }
  val io: MicroTageIO = IO(new MicroTageIO)
  io.resetDone   := true.B
  io.train.ready := true.B

  /* *** submodules *** */
  private val tables = TableInfos.zipWithIndex.map {
    case (info, i) =>
      val t = Module(new MicroTage1Table(
        numSets = info.NumSets,
        histLen = info.HistoryLength,
        tagLen = info.TagWidth,
        histBitsInTag = info.HistBitsInTag,
        tableId = i
      )).io
      t
  }
  private val tickCounter = RegInit(0.U((TickWidth + 1).W))
  // Predict
  tables.foreach { t =>
    t.req.startPc        := io.startVAddr
    t.req.foldedPathHist := io.foldedPathHist
    t.usefulReset        := tickCounter(TickWidth)
    t.usefulPenalty      := false.B
  }
  private val takenCases       = tables.reverse.map(t => t.resp.valid -> t.resp.bits.taken)
  private val cfiPositionCases = tables.reverse.map(t => t.resp.valid -> t.resp.bits.cfiPosition)
  private val usefulCase       = tables.reverse.map(t => t.resp.valid -> t.resp.bits.hitUseful)
  private val takenCtrCase     = tables.reverse.map(t => t.resp.valid -> t.resp.bits.hitTakenCtr)

  private val histTableHitMap         = tables.map(_.resp.valid)
  private val histTableTakenMap       = tables.map(_.resp.bits.taken)
  private val histTableUsefulVec      = VecInit(tables.map(_.resp.bits.useful))
  private val histTableCfiPositionVec = VecInit(tables.map(_.resp.bits.cfiPosition))
  private val choseTableTakenCtr      = MuxCase(0.U.asTypeOf(new SaturateCounter(TakenCtrWidth)), takenCtrCase)
  private val choseTableUseful        = MuxCase(0.U.asTypeOf(new SaturateCounter(UsefulWidth)), usefulCase)

  // ------ Base ----
  private val baseTable = Module(new MicroBaseTable1(BaseTableSize))
  baseTable.io.req.startPc          := io.startVAddr
  baseTable.io.update.valid         := io.train.valid
  baseTable.io.update.bits.startPc  := io.train.bits.startVAddr
  baseTable.io.update.bits.branches := io.train.bits.branches

  private val histTableHit         = tables.map(_.resp.valid).reduce(_ || _)
  private val histTableTaken       = MuxCase(false.B, takenCases)
  private val histTableCfiPosition = MuxCase(0.U(CfiPositionWidth.W), cfiPositionCases)

  private val finalPredTaken       = Mux(histTableHit, histTableTaken, baseTable.io.resp.taken)
  private val finalPredCfiPosition = Mux(histTableHit, histTableCfiPosition, baseTable.io.resp.cfiPosition)
  private val prediction           = Wire(new MicroTage1Prediction)
  prediction.taken                             := finalPredTaken && choseTableTakenCtr.isSaturatePositive
  prediction.cfiPosition                       := finalPredCfiPosition
  prediction.meta.valid                        := tables.map(_.resp.valid).reduce(_ || _)
  prediction.meta.bits.histTableHitMap         := tables.map(_.resp.valid)
  prediction.meta.bits.histTableHit            := tables.map(_.resp.valid).reduce(_ || _)
  prediction.meta.bits.histTableTakenMap       := tables.map(_.resp.bits.taken)
  prediction.meta.bits.histTableUsefulVec      := histTableUsefulVec
  prediction.meta.bits.histTableCfiPositionVec := histTableCfiPositionVec
  prediction.meta.bits.hitUseful               := choseTableUseful
  prediction.meta.bits.hitTakenCtr             := choseTableTakenCtr
  prediction.meta.bits.baseTaken               := baseTable.io.resp.taken
  prediction.meta.bits.baseCfiPosition         := baseTable.io.resp.cfiPosition
  prediction.meta.bits.taken                   := finalPredTaken
  prediction.meta.bits.cfiPosition             := finalPredCfiPosition
  io.prediction := RegEnable(prediction, 0.U.asTypeOf(new MicroTage1Prediction), io.stageCtrl.s0_fire)

  private val train   = Wire(Valid(new BpuTrain))
  train.valid := io.train.valid
  train.bits  := io.train.bits
  private val trainNext     = RegNext(train, 0.U.asTypeOf(Valid(new BpuTrain)))
  private val t1_trainData  = trainNext.bits
  private val t1_trainMeta  = trainNext.bits.meta.tempTage
  private val t1_trainValid = Wire(Bool())
  t1_trainValid := trainNext.valid

  // ------------ MicroTage is only concerned with conditional branches ---------- //
  private val t1_LT_taken_misPred = VecInit(
    t1_trainData.branches.map { b =>
      b.valid && b.bits.attribute.isConditional && (t1_trainMeta.taken && ((b.bits.cfiPosition < t1_trainMeta.cfiPosition) && b.bits.taken)) 
    }
  )
  private val t1_GT_taken_misPred = VecInit(
    t1_trainData.branches.map { b =>
      b.valid && b.bits.attribute.isConditional && (t1_trainMeta.taken && (b.bits.cfiPosition > t1_trainMeta.cfiPosition))
    }
  )
  private val t1_EQ_taken_misPred = VecInit(
    t1_trainData.branches.map { b =>
      b.valid && b.bits.attribute.isConditional && (t1_trainMeta.taken && (b.bits.cfiPosition === t1_trainMeta.cfiPosition) && !b.bits.taken)
    }
  )

  private val t1_LT_noTaken_misPred = VecInit(
    t1_trainData.branches.map { b =>
      b.valid && b.bits.attribute.isConditional && (!t1_trainMeta.taken && ((b.bits.cfiPosition < t1_trainMeta.cfiPosition) && b.bits.taken))
    }
  )
  private val t1_GT_noTaken_misPred = VecInit(
    t1_trainData.branches.map { b =>
      b.valid && b.bits.attribute.isConditional && (!t1_trainMeta.taken && (b.bits.cfiPosition > t1_trainMeta.cfiPosition) && b.bits.taken)
    }
  )
  private val t1_EQ_noTaken_misPred = VecInit(
    t1_trainData.branches.map { b =>
      b.valid && b.bits.attribute.isConditional && (!t1_trainMeta.taken && (b.bits.cfiPosition === t1_trainMeta.cfiPosition) && b.bits.taken)
    }
  )
  private val t1_misPred    = VecInit.tabulate(ResolveEntryBranchNumber){ i =>
    t1_LT_taken_misPred(i) || t1_GT_taken_misPred(i) || t1_EQ_taken_misPred(i) || t1_LT_noTaken_misPred(i) || t1_GT_noTaken_misPred(i) || t1_EQ_noTaken_misPred(i)
  }

  private val t1_hasPredBr  = VecInit(t1_trainData.branches.map(b =>
    b.valid && b.bits.attribute.isConditional && (b.bits.cfiPosition === t1_trainMeta.cfiPosition)
  ))
  private val t1_histUpdateBranch = Mux1H(t1_hasPredBr, t1_trainData.branches)

  private val t1_takenBranchVec = VecInit(t1_trainData.branches.map(b =>
    b.valid && b.bits.attribute.isConditional && b.bits.taken  
  ))
  private val t1_hasTaken       = t1_takenBranchVec.reduce(_ || _)
  private val t1_positions      = VecInit(t1_trainData.branches.map(_.bits.cfiPosition))
  private val t1_compareMatrix  = CompareMatrix(t1_positions)
  private val t1_minTakenOH     = t1_compareMatrix.getLeastElementOH(t1_takenBranchVec)
  private val t1_minTakenBranch = Mux1H(t1_minTakenOH, t1_trainData.branches)
  private val t1_allocCfiPosition = Mux(t1_hasTaken, t1_minTakenBranch.bits.cfiPosition, t1_trainMeta.cfiPosition)
  private val t1_allocTaken       = Mux(t1_hasTaken, true.B, !t1_trainMeta.taken)
  private val t1_updateCfiposition  = t1_trainMeta.cfiPosition

  private val t1_histTableNeedAlloc  = t1_misPred.reduce(_ || _)
  private val t1_histTableNeedUpdate = t1_hasPredBr.reduce(_ || _) && t1_trainMeta.histTableHit
  private val t1_histHitMisPred      = t1_misPred.reduce(_ || _) && t1_trainMeta.histTableHit
  private val t1_updateTaken         = t1_hasPredBr.reduce(_ || _) && (t1_histUpdateBranch.bits.taken)

  private val t1_providerMask = PriorityEncoderOH(t1_trainMeta.histTableHitMap.reverse).reverse
  private val t1_histTableNoUseful = t1_trainMeta.histTableUsefulVec.map(useful => useful === 0.U).asUInt
  private val t1_fastAllocMask   = t1_providerMask.asUInt & t1_histTableNoUseful
  private val hitMask            = t1_trainMeta.histTableHitMap.asUInt
  private val lowerFillMask      = Mux(hitMask === 0.U, 0.U, hitMask | (hitMask - 1.U))
  private val usefulMask         = t1_trainMeta.histTableUsefulVec.map(useful => useful(UsefulWidth - 1)).asUInt
  private val allocCandidateMask = ~(lowerFillMask | usefulMask)
  // private val t1_allocMask       = PriorityEncoderOH(allocCandidateMask)
  private val normalAllocMask      = PriorityEncoderOH(allocCandidateMask)
  private val t1_allocMask         = Mux(t1_fastAllocMask.orR, t1_fastAllocMask, normalAllocMask)

  when(tickCounter(TickWidth)) {
    tickCounter := 0.U
  }.elsewhen((t1_allocMask === 0.U) && t1_histTableNeedAlloc && t1_trainValid) {
    tickCounter := tickCounter + 1.U
  }

// The training logic consists of two operations: updating entries and
// allocating new ones.
//
// Update behavior:
// - If train_position < table_position: entry remains unchanged.
// - If train_position === table_position: value is adjusted based on
//   prediction outcome (increment or decrement).
// - If train_position > table_position: entry remains unchanged.
//
// Allocation behavior (triggered only on misprediction):
// - A new entry is allocated to a higher-level table.
// - Target the lowest such table with an available slot (useful == 0).
// - If no slot is available, allocation fails.
//
// Update rule: To reduce noise, updates occur only when positions match.
//              The direction (inc/dec) is determined by the training result.
//
// Allocation rule: The selected entry replaces an available slot.
//                  If no free slot exists, allocation fails.
//                  Each failure is recorded; after 8 consecutive failures,
//                  all 'useful' counters are reset to 0.

  private val t1_allowAlloc = true.B
  tables.zipWithIndex.foreach { case (t, i) =>
    t.update.valid := ((t1_allocMask(i) && t1_histTableNeedAlloc && t1_allowAlloc) ||
      (t1_providerMask(i) && t1_histTableNeedUpdate)) && t1_trainValid
    t.update.bits.startPc := t1_trainData.startVAddr
    t.update.bits.cfiPosition := Mux(
      t1_allocMask(i) && t1_histTableNeedAlloc,
      t1_allocCfiPosition,
      t1_updateCfiposition
    )
    t.update.bits.alloc                  := t1_allocMask(i) && t1_histTableNeedAlloc
    t.update.bits.allocTaken             := t1_allocTaken
    t.update.bits.correct                := !t1_histHitMisPred
    t.update.bits.taken                  := t1_updateTaken
    t.update.bits.foldedPathHistForTrain := io.foldedPathHistForTrain
    t.update.bits.oldTakenCtr            := t1_trainMeta.hitTakenCtr
    t.update.bits.oldUseful              := t1_trainMeta.hitUseful
  }

  // ==========================================================================
  // === PERF === Performance Counters Section
  // ==========================================================================
  private val t1_trainHasBr = VecInit(t1_trainData.branches.map(b =>
    b.valid && b.bits.attribute.isConditional
  )).reduce(_ || _)
  private val t1_hasMisPred = t1_misPred.reduce(_ || _)
  private val t1_misPredEQ = VecInit(t1_trainData.branches.map(b =>
    b.valid && b.bits.attribute.isConditional &&
      ((b.bits.cfiPosition === t1_trainMeta.cfiPosition) && (b.bits.taken ^ t1_trainMeta.taken)) &&
      t1_trainMeta.histTableHit
  )).reduce(_ || _)
  private val t1_misPredLT = VecInit(t1_trainData.branches.map(b =>
    b.valid && b.bits.attribute.isConditional &&
      ((b.bits.cfiPosition < t1_trainMeta.cfiPosition) && b.bits.taken) && t1_trainMeta.histTableHit
  )).reduce(_ || _)
  private val t1_misPredGT = VecInit(t1_trainData.branches.map(b =>
    b.valid && b.bits.attribute.isConditional &&
      (b.bits.cfiPosition > t1_trainMeta.cfiPosition) && t1_trainMeta.histTableHit
  )).reduce(_ || _)
  // === Training feedback stage ===
  XSPerfAccumulate("microtage_train_valid", t1_trainValid)
  XSPerfAccumulate("microtage_train_br_valid", t1_trainValid && t1_trainHasBr)
  XSPerfAccumulate("microtage_train_br_taken_valid", t1_trainValid && t1_trainHasBr && t1_hasTaken)
  XSPerfAccumulate("microtage_train_br_histHit", t1_trainValid && t1_trainHasBr && t1_trainMeta.histTableHit)

  // train hit and correct
  private val predBrCorrect = t1_trainValid && t1_hasPredBr.reduce(_ || _) && !t1_misPred.reduce(_ || _)
  private val predBrWrong   = t1_trainValid && t1_hasPredBr.reduce(_ || _) && t1_misPred.reduce(_ || _)
  XSPerfAccumulate("microtage_train_has_predBr_correct", predBrCorrect)
  XSPerfAccumulate("microtage_train_has_predBr_wrong", predBrWrong)
  XSPerfAccumulate("microtage_train_hasBr_misPred", t1_trainValid && t1_trainHasBr && t1_misPred.reduce(_ || _))
  XSPerfAccumulate("microtage_train_hasBr_correctPred", t1_trainValid && t1_trainHasBr && (!t1_misPred.reduce(_ || _)))

  XSPerfAccumulate("microtage_train_histHit_LT_predBr_wrong", t1_trainValid && t1_misPredLT)
  XSPerfAccumulate("microtage_train_histHit_EQ_predBr_wrong", t1_trainValid && t1_misPredEQ)
  XSPerfAccumulate("microtage_train_hitHit_GT_predBr_wrong", t1_trainValid && t1_misPredGT)

  for (i <- 0 until NumTables) {
    XSPerfAccumulate(
      s"microtage_train_hit_EQ_predBr_multiHIt_counter${i + 1}",
      t1_trainValid && t1_hasPredBr.reduce(_ || _) && (PopCount(t1_trainMeta.histTableHitMap) === (i + 1).U)
    )
    XSPerfAccumulate(
      s"microtage_train_hit_EQ_predBr_multiHit_wrong_counter${i + 1}",
      t1_trainValid && t1_hasPredBr.reduce(_ || _) && t1_hasMisPred && (PopCount(
        t1_trainMeta.histTableHitMap
      ) === (i + 1).U)
    )
  }

  // === PHR Test ===
  private val testIdxFhInfos = TableInfos.zipWithIndex.map {
    case (info, i) =>
      val t = new FoldedHistoryInfo(info.HistoryLength, min(log2Ceil(info.NumSets), info.HistoryLength))
      t
  }
  private val testTagFhInfos = TableInfos.zipWithIndex.map {
    case (info, i) =>
      val t = new FoldedHistoryInfo(info.HistoryLength, min(info.HistoryLength, info.HistBitsInTag))
      t
  }
  private val testAltTagFhInfos = TableInfos.zipWithIndex.map {
    case (info, i) =>
      val t = new FoldedHistoryInfo(info.HistoryLength, min(info.HistoryLength, info.HistBitsInTag - 1))
      t
  }

  def computeHash(startPc: UInt, allFh: PhrAllFoldedHistories, tableId: Int): (UInt, UInt) = {
    val unhashedIdx = getUnhashedIdx(startPc)
    val unhashedTag = getUnhashedTag(startPc)
    val idxFh       = allFh.getHistWithInfo(testIdxFhInfos(tableId)).foldedHist
    val tagFh       = allFh.getHistWithInfo(testTagFhInfos(tableId)).foldedHist
    val altTagFh    = allFh.getHistWithInfo(testAltTagFhInfos(tableId)).foldedHist
    val idx = if (testIdxFhInfos(tableId).FoldedLength < log2Ceil(TableInfos(tableId).NumSets)) {
      (unhashedIdx ^ Cat(idxFh, idxFh))(log2Ceil(TableInfos(tableId).NumSets) - 1, 0)
    } else {
      (unhashedIdx ^ idxFh)(log2Ceil(TableInfos(tableId).NumSets) - 1, 0)
    }
    val lowTag  = (unhashedTag ^ tagFh ^ (altTagFh << 1))(TableInfos(tableId).HistBitsInTag - 1, 0)
    val highTag = connectPcTag(unhashedIdx, tableId)
    val tag     = Cat(highTag, lowTag)(TableInfos(tableId).TagWidth - 1, 0)
    (idx, tag)
  }

  private val (s0_idxTable0, s0_tagTable0) = computeHash(io.startVAddr.toUInt, io.foldedPathHist, 0)
  private val (s0_idxTable1, s0_tagTable1) = computeHash(io.startVAddr.toUInt, io.foldedPathHist, 1)

  prediction.meta.bits.testPredIdx0 := s0_idxTable0
  prediction.meta.bits.testPredTag0 := s0_tagTable0
  prediction.meta.bits.testPredIdx1 := s0_idxTable1
  prediction.meta.bits.testPredTag1 := s0_tagTable1

  prediction.meta.bits.testPredStartAddr := io.startVAddr.toUInt

  private val (trainIdx0, trainTag0) = computeHash(t1_trainData.startVAddr.toUInt, io.foldedPathHistForTrain, 0)
  private val (trainIdx1, trainTag1) = computeHash(t1_trainData.startVAddr.toUInt, io.foldedPathHistForTrain, 1)

  XSPerfAccumulate("train_idx_hit", t1_trainValid && (t1_trainMeta.testPredIdx0 === trainIdx0))
  XSPerfAccumulate("train_tag_hit", t1_trainValid && (t1_trainMeta.testPredTag0 === trainTag0))
  XSPerfAccumulate("train_idx_miss", t1_trainValid && (t1_trainMeta.testPredIdx0 =/= trainIdx0))
  XSPerfAccumulate("train_tag_miss", t1_trainValid && (t1_trainMeta.testPredTag0 =/= trainTag0))

  t1_trainValid := trainNext.valid && (t1_trainMeta.testPredIdx0 === trainIdx0) && (t1_trainMeta.testPredTag0 === trainTag0) &&
    (t1_trainMeta.testPredIdx1 === trainIdx1) && (t1_trainMeta.testPredTag1 === trainTag1)

}
