use std::ops::Deref;

use ndarray::{Array2, Array3, s};

use crate::powerset::PowersetMapping;

use super::{ChunkLayout, PipelineError};

pub(in crate::pipeline) struct PendingEmbedding<'a> {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub audio: &'a [f32],
    pub mask: Vec<f32>,
    pub clean_mask: Vec<f32>,
}

pub(in crate::pipeline) struct PendingSplitEmbedding {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub fbank_idx: usize,
    pub weights: Vec<f32>,
}

/// Decoded powerset segmentations per chunk, shape (chunks, frames, speakers)
#[derive(Debug, Clone)]
pub struct DecodedSegmentations(pub Array3<f32>);

impl Deref for DecodedSegmentations {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Speaker embeddings per chunk, shape (chunks, speakers, embedding_dim)
#[derive(Debug, Clone)]
pub struct ChunkEmbeddings(pub Array3<f32>);

impl Deref for ChunkEmbeddings {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Number of active speakers per chunk
#[derive(Debug, Clone)]
pub struct SpeakerCountTrack(pub Vec<usize>);

impl Deref for SpeakerCountTrack {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Cluster assignments per chunk-speaker pair, shape (chunks, speakers)
///
/// Values are cluster IDs (-1 for unassigned)
#[derive(Debug, Clone)]
pub struct ChunkSpeakerClusters(pub Array2<i32>);

impl Deref for ChunkSpeakerClusters {
    type Target = Array2<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Frame-level binary speaker activations, shape (frames, speakers)
#[derive(Debug, Clone)]
pub struct DiscreteDiarization(pub Array2<f32>);

impl Deref for DiscreteDiarization {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DiscreteDiarization {
    /// Zero out all but the highest-scoring speaker in each frame, making activations exclusive
    pub fn make_exclusive(&mut self) {
        crate::reconstruct::make_exclusive(&mut self.0);
    }

    /// Convert frame activations to time-stamped speaker segments using default frame timing.
    pub fn to_segments(&self) -> Vec<crate::segment::Segment> {
        self.to_segments_with(
            crate::pipeline::FRAME_STEP_SECONDS,
            crate::pipeline::FRAME_DURATION_SECONDS,
        )
    }

    /// Convert frame activations to time-stamped speaker segments using custom frame timing.
    pub fn to_segments_with(
        &self,
        frame_step_seconds: f64,
        frame_duration_seconds: f64,
    ) -> Vec<crate::segment::Segment> {
        crate::segment::to_segments(&self.0, frame_step_seconds, frame_duration_seconds)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FrameActivations(pub(crate) Array2<f32>);

impl Deref for FrameActivations {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(in crate::pipeline) struct RawSegmentationWindows(pub Vec<Array2<f32>>);

impl RawSegmentationWindows {
    pub(in crate::pipeline) fn decode(self, powerset: &PowersetMapping) -> DecodedSegmentations {
        let mut windows = self.0.into_iter();
        let Some(first_window) = windows.next() else {
            return DecodedSegmentations(Array3::zeros((0, 0, 0)));
        };

        let num_windows = windows.len() + 1;
        let first = powerset.hard_decode(&first_window);
        let mut stacked = Array3::<f32>::zeros((num_windows, first.nrows(), first.ncols()));
        stacked.slice_mut(s![0, .., ..]).assign(&first);

        for (window_idx, window) in windows.enumerate() {
            let decoded = powerset.hard_decode(&window);
            stacked
                .slice_mut(s![window_idx + 1, .., ..])
                .assign(&decoded);
        }

        DecodedSegmentations(stacked)
    }
}

/// Input for batch diarization
pub struct BatchInput<'a> {
    /// Mono 16kHz audio samples
    pub audio: &'a [f32],
    /// Identifier used in RTTM output lines
    pub file_id: &'a str,
}

/// Intermediate results from segmentation and embedding inference
pub struct InferenceArtifacts {
    pub(in crate::pipeline) layout: ChunkLayout,
    pub(in crate::pipeline) segmentations: DecodedSegmentations,
    pub(in crate::pipeline) embeddings: ChunkEmbeddings,
}

impl InferenceArtifacts {
    /// bq-Meeting PoC ローカル改変: chunk 分割で個別に走らせた `run_inference_only` の
    /// 結果をチャンク軸（axis 0）で連結し、1回の `run_post_inference` に渡せる形にする。
    /// `layout` は先頭パートの layout に `with_num_chunks(合計チャンク数)` を掛けたもの。
    /// frames / speakers / embedding_dim の形状（axis 1, 2）が全パートで一致しない場合や
    /// `parts` が空の場合はエラーを返す。
    pub fn concat(parts: Vec<InferenceArtifacts>) -> Result<InferenceArtifacts, PipelineError> {
        let mut parts = parts.into_iter();
        let Some(first) = parts.next() else {
            return Err(PipelineError::Invariant(
                "InferenceArtifacts::concat: parts is empty".to_owned(),
            ));
        };

        let seg_tail_shape = first.segmentations.0.shape()[1..].to_vec();
        let emb_tail_shape = first.embeddings.0.shape()[1..].to_vec();

        let mut total_chunks = first.segmentations.0.shape()[0];
        let mut seg_arrays = vec![first.segmentations.0];
        let mut emb_arrays = vec![first.embeddings.0];

        for part in parts {
            let part_seg_tail = part.segmentations.0.shape()[1..].to_vec();
            let part_emb_tail = part.embeddings.0.shape()[1..].to_vec();
            if part_seg_tail != seg_tail_shape || part_emb_tail != emb_tail_shape {
                return Err(PipelineError::Invariant(format!(
                    "InferenceArtifacts::concat: shape mismatch (segmentations {:?} vs {:?}, embeddings {:?} vs {:?})",
                    part_seg_tail, seg_tail_shape, part_emb_tail, emb_tail_shape
                )));
            }
            total_chunks += part.segmentations.0.shape()[0];
            seg_arrays.push(part.segmentations.0);
            emb_arrays.push(part.embeddings.0);
        }

        let seg_views: Vec<_> = seg_arrays.iter().map(|arr| arr.view()).collect();
        let emb_views: Vec<_> = emb_arrays.iter().map(|arr| arr.view()).collect();

        let segmentations = ndarray::concatenate(ndarray::Axis(0), &seg_views).map_err(|err| {
            PipelineError::Invariant(format!(
                "InferenceArtifacts::concat: segmentations concat failed: {err}"
            ))
        })?;
        let embeddings = ndarray::concatenate(ndarray::Axis(0), &emb_views).map_err(|err| {
            PipelineError::Invariant(format!(
                "InferenceArtifacts::concat: embeddings concat failed: {err}"
            ))
        })?;

        let layout = first.layout.with_num_chunks(total_chunks);

        Ok(InferenceArtifacts {
            layout,
            segmentations: DecodedSegmentations(segmentations),
            embeddings: ChunkEmbeddings(embeddings),
        })
    }

    /// bq-Meeting PoC ローカル改変: chunk 分割で重なり（オーバーラップ）を持たせて
    /// `run_inference_only` した結果から、末尾の余剰 window（オーバーラップ分・末尾の
    /// ゼロ詰め window）を切り落とす。axis 0（チャンク軸）の先頭 `num_chunks` 個だけを残す
    /// （segmentations・embeddings 両方）。`num_chunks` が現在のチャンク数以上なら、
    /// 実質的にそのまま返す（`with_num_chunks` の再計算だけが起きる）。
    pub fn truncate_chunks(self, num_chunks: usize) -> InferenceArtifacts {
        let current_chunks = self.segmentations.0.shape()[0];
        let keep = num_chunks.min(current_chunks);

        let segmentations = self.segmentations.0.slice(s![0..keep, .., ..]).to_owned();
        let embeddings = self.embeddings.0.slice(s![0..keep, .., ..]).to_owned();
        let layout = self.layout.with_num_chunks(keep);

        InferenceArtifacts {
            layout,
            segmentations: DecodedSegmentations(segmentations),
            embeddings: ChunkEmbeddings(embeddings),
        }
    }
}

/// Complete output from a diarization run
pub struct DiarizationResult {
    /// Decoded segmentations from the powerset model
    pub segmentations: DecodedSegmentations,
    /// Speaker embeddings extracted from each chunk
    pub embeddings: ChunkEmbeddings,
    /// Number of active speakers per chunk
    pub speaker_count: SpeakerCountTrack,
    /// Cluster assignment for each chunk-speaker pair
    pub hard_clusters: ChunkSpeakerClusters,
    /// Frame-level binary speaker activations after reconstruction
    pub discrete_diarization: DiscreteDiarization,
    /// Merged speaker segments (time-stamped speaker turns)
    pub segments: Vec<crate::segment::Segment>,
}

impl DiarizationResult {
    /// Render RTTM output with the given file identifier
    pub fn rttm(&self, file_id: &str) -> String {
        crate::segment::to_rttm(&self.segments, file_id)
    }
}

pub(in crate::pipeline) enum InferencePath {
    Sequential,
    Concurrent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::pipeline) enum EmbeddingPath {
    Masked,
    Split,
    MultiMask,
}
