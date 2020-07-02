use ndarray::prelude::*;

use ndarray_rand::{RandomExt, F32};
use ndarray_rand::rand_distr::Uniform;

use crate::activation::DerivableLayer;

pub struct Layer{
    weights: Array2<f32>,
    biases: Array1<f32>,
    input: Array1<f32>,
    output: Array1<f32>,
    activation: Box<dyn DerivableLayer>
}

impl Layer {
    // Constructor
    pub fn new(in_features: u16, out_features: u16, activation_fnc: Box<dyn DerivableLayer>) -> Layer {
        let dist = Uniform::new(-1., 1.);
        Layer{
            weights: Array::random([in_features as usize, out_features as usize], F32(dist)),
            biases: Array::random(out_features as usize, F32(dist)),
            input: Array::zeros(in_features as usize),
            output: Array::zeros(out_features as usize),
            activation: activation_fnc
        }
    }

    pub fn forward(&mut self, mut arr: Array1<f32>) -> Array1<f32> {
        self.input.assign(&arr.clone());
        arr = arr.dot(&self.weights);
        arr += &self.biases;
        self.output.assign(&arr.clone()); // we keep the layer output before activation
        arr = self.activation.forward(arr);
        arr
    }


    pub fn backward(&mut self, delta: &Array2<f32>, l_rate: &f32) -> Array2<f32> {
        // println!("delta: {:?} ~~ weights: {:?} ~~ biases: {}", delta.dim(), self.weights.dim(), self.biases.dim());
        let newdelta = delta * &self.activation.backward(self.output.clone()).insert_axis(Axis(1));
        // println!("newdelta: {:?}", newdelta.mean());
        let weights_delta = self.input.clone().insert_axis(Axis(1)).dot(&newdelta.t());
        
        // println!("input: {:?}", self.input.dim());
        // println!("newdelta: {:?}", newdelta.dim());
        // println!("weights_delta: {:?}", weights_delta.dim());

        let previous_layer_delta = self.weights.dot(&newdelta);
        // println!("previous_layer_delta: {:?}", previous_layer_delta.dim());

        self.weights += &(&weights_delta * l_rate.clone());
        self.biases  += &(&newdelta.column(0) * l_rate.clone());
        return previous_layer_delta
    }

}

