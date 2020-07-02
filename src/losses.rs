
use ndarray::prelude::*;

pub trait DerivableLoss {
    fn forward(&self, pred: &Array1<f32>, ground_truth: &Array1<f32>) -> f32;
    fn backward(&self, pred: &Array1<f32>, ground_truth: &Array1<f32>) -> Array1<f32>;
}

pub struct RMS;

impl DerivableLoss for RMS {
    fn forward(&self, pred: &Array1<f32>, ground_truth: &Array1<f32>) -> f32 {
        let diff: Array1<f32> = (ground_truth.clone() - pred).mapv(|x|{x.powf(2.)});
        diff.sum()
    }

    fn backward(&self, pred: &Array1<f32>, ground_truth: &Array1<f32>) -> Array1<f32> {
        ground_truth - pred
    }
}
