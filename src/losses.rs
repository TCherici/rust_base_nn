
use ndarray::prelude::*;

pub fn rms(pred: Array1<f32>, ground_truth: Array1<f32>) -> f32 {
    let diff: Array1<f32> = ground_truth - pred;
    diff.sum()
}