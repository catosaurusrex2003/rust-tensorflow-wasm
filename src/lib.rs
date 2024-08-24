use wasm_bindgen::prelude::*;

// Import necessary modules and dependencies
use std::error::Error;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Status, Tensor};

// Expose the function to JavaScript
#[wasm_bindgen]
pub fn run_tensorflow_model(x_val: i32, y_val: i32, model_data: &[u8]) -> Result<i32, JsValue> {
    // Handling the errors in a way compatible with wasm
    match run_model_internal(x_val, y_val, model_data) {
        Ok(result) => Ok(result),
        Err(e) => Err(JsValue::from_str(&format!("Error: {}", e))),
    }
}

fn run_model_internal(x_val: i32, y_val: i32, model_data: &[u8]) -> Result<i32, Box<dyn Error>> {
    // Create input variables for our addition
    let mut x = Tensor::new(&[1]);
    x[0] = x_val;
    let mut y = Tensor::new(&[1]);
    y[0] = y_val;

    // Load the computation graph from the provided model data
    let mut graph = Graph::new();
    graph.import_graph_def(model_data, &ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("x")?, 0, &x);
    args.add_feed(&graph.operation_by_name_required("y")?, 0, &y);
    let z = args.request_fetch(&graph.operation_by_name_required("z")?, 0);
    session.run(&mut args)?;

    // Check our results.
    let z_res: i32 = args.fetch(z)?[0];
    Ok(z_res)
}
