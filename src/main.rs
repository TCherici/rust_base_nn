mod nnet;
mod activation;
mod layers;
mod losses;

use ndarray::{array};
use ndarray::prelude::*;
use nnet::NNet;

use rust_mnist::{print_sample_image, Mnist};
use rand::seq::SliceRandom;
use rand::thread_rng;

const INPUT_SIZE: u16 = 28 * 28;
const OUTPUT_SIZE: u16 = 10;

fn main() {
    
    let nnet_topology: Array1<u16> = array![128, 64, 32, OUTPUT_SIZE];
    let mut nnet: NNet = nnet::NNet::new(INPUT_SIZE, nnet_topology);
   
    let mnist = Mnist::new("/home/tcherici/Documents/mnist/");

    // Print one image (the one at index 3) for verification.
    print_sample_image(&mnist.train_data[3], mnist.train_labels[3]);

    println!("====== TRAINING =======");
    let mut idxs: Vec<u32> = (0..(mnist.train_data.len() as u32)).collect();
    let mut rng = thread_rng();
    let mut count: u16 = 0;
    let mut rmse_sum: f32 = 0.;
    idxs.shuffle(&mut rng);
    for idx in idxs.iter() {
        count += 1;
        let input_f32: Array1<f32> = mnist
            .train_data[*idx as usize]
            .iter()
            .map(|&x| {x as f32 / 255.})
            .collect();
        let mut gt_array: Array1<f32> = Array::zeros(OUTPUT_SIZE as usize);
        let gt_val: &u8 = &mnist.train_labels[*idx as usize];
        gt_array[*gt_val as usize] = 1.;
        let output = nnet.forward(input_f32.clone());
        nnet.backward(&output, &gt_array);
        rmse_sum += nnet.get_loss(&output, &gt_array);
        if count % 100 == 0 {
            println!("avg rmse: {:.3}", rmse_sum/100.);
            rmse_sum = 0.
        }
        if count % 1000 == 0{
            nnet.l_rate *= 0.5;
            println!("Iteration {} || lr: {}", count, nnet.l_rate);
        }
    }

    println!("====== TESTING =======");
    let mut rmse_sum: f32 = 0.;
    let no_test_images = mnist.test_data.len();
    for idx in 0..no_test_images{
        let input_u8 = &mnist.test_data[idx as usize];
        let input_f32: Array1<f32> = input_u8.iter().map(|&x| {x as f32 / 255.}).collect();
        let output = nnet.forward(input_f32.clone());
        let mut gt_array: Array1<f32> = Array::zeros(OUTPUT_SIZE as usize);
        let gt_val: &u8 = &mnist.train_labels[idx as usize];
        gt_array[*gt_val as usize] = 1.;
        rmse_sum += nnet.get_loss(&output, &gt_array);
    }
    rmse_sum /= no_test_images as f32;
    println!("Final RMSE: {}", rmse_sum);

}
