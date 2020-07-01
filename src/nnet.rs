extern crate ndarray;

use ndarray::prelude::*;

use crate::activation;
use crate::layers;
use crate::losses;
use layers::Layer;

use crate::activation::{Derivable, Sigmoid, ReLU};

pub struct NNet{
    input_size: u16,
    layers: Vec<Layer>,
    loss_func: fn(Array1<f32>, Array1<f32>) -> f32
}

impl NNet {
    // Constructor
    pub fn new(input_size: u16, topology: Array1::<u16>) -> NNet {
        let mut layers: Vec<Layer> = vec![];
        for idx in 0..topology.len() {
            if idx == 0 {
                layers.push(Layer::new(input_size, topology[idx], Box::new(activation::ReLU)))
            } else if idx == topology.len() - 1{
                layers.push(Layer::new(topology[idx - 1], topology[idx], Box::new(activation::Sigmoid)))
            } else {
                layers.push(Layer::new(topology[idx - 1], topology[idx], Box::new(activation::ReLU)))
            }
        }

        NNet{
            input_size: input_size,
            layers: layers,
            loss_func: losses::rms
        }
    }

    pub fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        let mut output: Array1<f32> = input.to_owned();
        for layer in &mut self.layers {
            output = layer.forward(output.clone())
        }
        output
    }

    pub fn calc_error(&self, ground_truth: Array1<f32>) {
        
    }

}
