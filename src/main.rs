mod nnet;
mod activation;
mod layers;
mod losses;

use ndarray::{array};
use ndarray::prelude::*;
use nnet::NNet;

use rust_mnist::{print_sample_image, Mnist};


fn main() {
    println!("Hello, world!");
    let input_size: u16 = 28 * 28;
    let nnet_topology: Array1<u16> = array![64, 64, 10];
    let mut nnet: NNet = nnet::NNet::new(input_size, nnet_topology);
   
    let mnist = Mnist::new("/home/tcherici/Documents/mnist/");

    // Print one image (the one at index 5) for verification.
    print_sample_image(&mnist.train_data[5], mnist.train_labels[5]);


    for idx in 1..mnist.train_data.len() as u16 {
        let input_u8 = &mnist.train_data[idx as usize];
        let input_f32: Array1<f32> = input_u8.iter().map(|&x| {x as f32 / 255.}).collect();
        let mut output_array: Array1<f32> = Array::zeros(10);
        let output_val: &u8 = &mnist.train_labels[idx as usize];
        output_array[*output_val as usize] = 1.;
        let output = nnet.forward(input_f32.clone());
        nnet.backward(&output, &output_array)
    }
    // let testarr: Array1<f32> = Array::random(28*28 as usize, F32(dist));
    // let testarr: Array1<f32> = array!(&mnist.test_data[0].iter().map(|x| {*x as f32}).collect());
    // println!("nnet forward: {}", nnet.forward(testarr));
}
