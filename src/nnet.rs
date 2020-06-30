extern crate ndarray;

use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::{RandomExt, F32};
use ndarray_rand::rand_distr::Uniform;


pub struct NNet {
    input_size: u16,
    layers: Vec<Layer>,
    deltas: Vec<f32>
}

impl NNet {
    // Constructor
    pub fn new(input_size: u16, topology: Array1<u16>) -> NNet {
        let mut layers: Vec<Layer> = vec![];
        for idx in 0..topology.len() {
            if idx == 0 {
                layers.push(Layer::new(input_size, topology[idx]))
            } else {
                layers.push(Layer::new(topology[idx - 1], topology[idx]))
            }
        }

        NNet{
            input_size: input_size,
            layers: layers,
            deltas: vec![0.]
        }
    }

    pub fn display(&self) -> String {
        format!("{}", self.layers[0].weights)
    }
}


pub struct Layer{
    weights: Array2<f32>,
    biases: Array1<f32>
}

impl Layer {
    // Constructor
    pub fn new(in_features: u16, out_features: u16) -> Layer {
        let dist = Uniform::new(-1., 1.);
        Layer {
            weights: Array::random([in_features as usize, out_features as usize], F32(dist)),
            biases: Array::random(out_features as usize, F32(dist))
        }
    }
}

