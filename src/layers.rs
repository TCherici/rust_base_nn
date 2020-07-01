use ndarray::prelude::*;

use ndarray_rand::{RandomExt, F32};
use ndarray_rand::rand_distr::Uniform;

use crate::activation::Derivable;

pub struct Layer{
    weights: Array2<f32>,
    biases: Array1<f32>,
    cache: Array1<f32>,
    activation: Box<dyn Derivable>
}

impl Layer {
    // Constructor
    pub fn new(in_features: u16, out_features: u16, activation_fnc: Box<dyn Derivable>) -> Layer {
        let dist = Uniform::new(-1., 1.);
        Layer{
            weights: Array::random([in_features as usize, out_features as usize], F32(dist)),
            biases: Array::random(out_features as usize, F32(dist)),
            cache: Array::zeros(out_features as usize),
            activation: activation_fnc
        }
    }

    pub fn forward(&mut self, mut output: Array1<f32>) -> Array1<f32> {
        output = output.dot(&self.weights);
        output += &self.biases;
        output = self.activation.forward(output);
        self.cache.assign(&output);
        output
    }

}

