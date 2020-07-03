use ndarray::prelude::*;

pub trait DerivableLayer {
    fn forward(&self, arr: Array1<f32>) -> Array1<f32>;
    fn backward(&self, arr: Array1<f32>) -> Array2<f32>;
}

pub struct ReLU;


impl DerivableLayer for ReLU {
    fn forward(&self, arr: Array1<f32>) -> Array1<f32> {
        arr.mapv(|x|{if x > 0. {x} else {0.}})
    }

    fn backward(&self, arr: Array1<f32>) -> Array2<f32> {
        arr.mapv(|x|{if x > 0. {1.} else {0.}}).insert_axis(Axis(1))
    }
    
}

pub struct Sigmoid;

impl DerivableLayer for Sigmoid {
    fn forward(&self, arr: Array1<f32>) -> Array1<f32> {
        arr.mapv(|x|{1./(1. + std::f32::consts::E.powf(-x))})
    }

    fn backward(&self, arr: Array1<f32>) -> Array2<f32> {
        // arr.mapv(|x|{std::f32::consts::E.powf(-x)/(1. + std::f32::consts::E.powf(-x)).powf(2.)})
        (self.forward(arr.clone()) * (1. - self.forward(arr))).insert_axis(Axis(1))
    }

}

pub struct SoftMax;

impl DerivableLayer for SoftMax {
    fn forward(&self, arr: Array1<f32>) -> Array1<f32> {
        // arr -= f32::max(arr);
        let sum: f32 = arr.mapv(|x|{std::f32::consts::E.powf(x)}).sum();
        arr.mapv(|x|{std::f32::consts::E.powf(x)/sum})
    }

    fn backward(&self, arr: Array1<f32>) -> Array2<f32> {
        let s: Array1<f32> = self.forward(arr);
        let s_diag: Array2<f32> = Array2::from_diag(&s);
        let s_matrix: Array2<f32> = s.clone().insert_axis(Axis(1)).dot(&s.insert_axis(Axis(0)));
        s_diag - &s_matrix * &s_matrix.t()
    }

}