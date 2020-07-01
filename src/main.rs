mod nnet;
mod activation;
mod layers;
mod losses;

use ndarray::prelude::*;
use nnet::NNet;

fn main() {
    println!("Hello, world!");
    let input_size: u16 = 28 * 28;
    let nnet_topology: Array1<u16> = array![64, 64, 10];
    let mut nnet: NNet = nnet::NNet::new(input_size, nnet_topology);

    let mut testarr: Array1<f32> = Array::ones(28*28);
    testarr[1] = -5.;

    println!("{}", nnet.forward(testarr));
}
