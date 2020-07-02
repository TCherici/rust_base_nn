mod nnet;
mod activation;
mod layers;
mod losses;

use ndarray_rand::{RandomExt, F32};
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use nnet::NNet;

fn main() {
    println!("Hello, world!");
    let input_size: u16 = 28 * 28;
    let nnet_topology: Array1<u16> = array![32, 64, 10];
    let mut nnet: NNet = nnet::NNet::new(input_size, nnet_topology);
    let dist = Uniform::new(0., 1.);
    // testarr[1] = -5.;

    let test1: Array2<f32> = Array::ones((64, 10));
    let test2: Array2<f32> = Array::ones((10, 1));
    println!("mmult: {:?}", (test1.dot(&test2)).dim());

    let mut gt : Array1<f32>  = Array::zeros(10);
    gt[6] = 1.;

    for _idx in 1..1600 {
        let testarr: Array1<f32> = Array::random(28*28 as usize, F32(dist));
        let output = nnet.forward(testarr.clone());
        // println!("nnet forward: {}", output);
        nnet.backward(&output, &gt)
    }
    let testarr: Array1<f32> = Array::random(28*28 as usize, F32(dist));
    println!("nnet forward: {}", nnet.forward(testarr.clone()));
}
