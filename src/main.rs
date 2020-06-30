mod nnet;

use ndarray::prelude::*;
use nnet::NNet;

fn main() {
    println!("Hello, world!");
    let input_size: u16 = 28 * 28;
    let nnet_topology: Array1<u16> = array![5, 3, 2];
    let nnet: NNet = nnet::NNet::new(input_size, nnet_topology);
    println!("{}", nnet.display())
}
