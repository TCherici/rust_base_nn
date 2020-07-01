use ndarray::prelude::*;

pub fn relu(arr: Array1<f32>) -> Array1<f32> {
    arr.mapv(|x|{if x > 0. {x} else {0.}})
}


pub fn sigmoid(arr: Array1<f32>) -> Array1<f32> {
    arr.mapv(|x|{1./(1. + std::f32::consts::E.powf(-x))})
}