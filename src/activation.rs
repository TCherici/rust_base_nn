use ndarray::prelude::*;

pub trait Derivable {
    fn forward(&self, arr: Array1<f32>) -> Array1<f32>;
    fn backward(&self, arr: Array1<f32>) -> Array1<f32>;
}

pub struct ReLU;


impl Derivable for ReLU {
    fn forward(&self, arr: Array1<f32>) -> Array1<f32> {
        arr.mapv(|x|{if x > 0. {x} else {0.}})
    }

    fn backward(&self, arr: Array1<f32>) -> Array1<f32> {
        arr.mapv(|x|{if x > 0. {1.} else {0.}})
    }
    
}

pub struct Sigmoid;

impl Derivable for Sigmoid {
    fn forward(&self, arr: Array1<f32>) -> Array1<f32> {
        arr.mapv(|x|{1./(1. + std::f32::consts::E.powf(-x))})
    }

    fn backward(&self, arr: Array1<f32>) -> Array1<f32> {
        arr.mapv(|x|{std::f32::consts::E.powf(-x)/(1. + std::f32::consts::E.powf(-x)).powf(2.)})
    }

}