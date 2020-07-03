extern crate ndarray;

use ndarray::prelude::*;

use crate::activation;
use crate::layers;
use crate::losses;
use layers::Layer;


pub struct NNet{
    layers: Vec<Layer>,
    loss_func: Box<dyn losses::DerivableLoss>,
    pub l_rate: f32
}

impl NNet {
    // Constructor
    pub fn new(input_size: u16, topology: Array1::<u16>) -> NNet {
        let mut layers: Vec<Layer> = vec![];
        for idx in 0..topology.len() {
            if idx == 0 {
                layers.push(Layer::new(input_size, topology[idx], Box::new(activation::Sigmoid)))
            } else if idx == topology.len() - 1{
                layers.push(Layer::new(topology[idx - 1], topology[idx], Box::new(activation::Sigmoid)))
            } else {
                layers.push(Layer::new(topology[idx - 1], topology[idx], Box::new(activation::Sigmoid)))
            }
        }

        NNet{
            layers: layers,
            loss_func: Box::new(losses::RMS),
            l_rate: 0.1
        }
    }

    pub fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        let mut output: Array1<f32> = input.to_owned();
        for layer in &mut self.layers {
            output = layer.forward(output)
        }
        output
    }

    pub fn get_loss(&self, pred: &Array1<f32>, ground_truth: &Array1<f32>) -> f32 {
        self.loss_func.forward(pred, ground_truth)
    }

    pub fn calc_error(&self, pred: &Array1<f32>, ground_truth: &Array1<f32>) -> Array2<f32> {
        let loss = self.loss_func.backward(pred, ground_truth);
        loss.insert_axis(Axis(1))
    }

    pub fn backward(&mut self, pred: &Array1<f32>, ground_truth: &Array1<f32>){
        let mut delta = self.calc_error(pred, ground_truth);
        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[layer_idx];
            delta = layer.backward(&delta, &self.l_rate);
            if layer_idx==self.layers.len() -1 {
            }
        }
    }

}
