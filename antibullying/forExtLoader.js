import * as tf from '@tensorflow/tfjs';

/**
 * Test whether a given URL is retrievable.
 */
export async function urlExists(url) {
  try {
    const response = await fetch(url, {method: 'HEAD'});
    console.log('Found URL.');
    return response.ok;
  } catch (err) {
    console.log(url + " = url doesn't exist");
    return false;
  }
}

/**
 * Load pretrained model stored at a remote URL.
 *
 * @return An instance of `tf.Model` with model topology and weights loaded.
 */
export async function loadHostedPretrainedModel(url) {
    console.log('Loading pretrained model from ' + url);
    try {
      const model = await tf.loadModel(url);
      console.log('Done loading model.');
      return model;
    } catch (err) {
      console.log("Failed to load model from " + url);
      console.error(err);
        
    }
  }

/**
 * Load metadata file stored at a remote URL.
 *
 * @return An object containing metadata as key-value pairs.
 */
export async function loadHostedMetadata(url) {
    console.log('Loading metadata from ' + url);
    try {
      const metadataJson = await fetch(url);
      const metadata = await metadataJson.json();
      console.log('Done loading metadata.');
      return metadata;
    } catch (err) {
      console.error(err);
      console.log('Loading metadata failed.');
    }
  }