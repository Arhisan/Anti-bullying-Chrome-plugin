import * as loader from './forExtLoader';
import * as tf from '@tensorflow/tfjs';

const HOSTED_URLS = {
    model: 'https://drive.google.com/file/d/1KetpFhi8hcF5V01VvkmfS8C_nAs8zgy8/view?usp=sharing',
    metadata: 'https://drive.google.com/file/d/1BfHohlw8Iu8eoKPfMPkfUXvRyDgh3xhq/view?usp=sharing'
};
  
const LOCAL_URLS = {
    model: 'http://localhost:1235/resources/model.json',
    metadata: 'http://localhost:1235/resources/metadata.json'
};


  export default class SentimentPredictor {
    /**
     * Initializes the Sentiment demo.
     */
    async init(urls) {
      this.urls = urls;
      this.model = await loader.loadHostedPretrainedModel(urls.model);
      await this.loadMetadata();
      return this;
    }
  
    async loadMetadata() {
      const sentimentMetadata =
          await loader.loadHostedMetadata(this.urls.metadata);
      this.indexFrom = 1//sentimentMetadata['index_from'];
      this.maxLen = sentimentMetadata['max_len'];
      console.log('indexFrom = ' + this.indexFrom);
      console.log('maxLen = ' + this.maxLen);
  
      this.wordIndex = sentimentMetadata['word_index']
    }
  
    predict(text) {
      // Convert to lower case and remove all punctuations.
      const inputText = text.split(' ');
          // text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
      // Look up word indices.
      const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
      for (let i = 0; i < inputText.length; ++i) {
        // TODO(cais): Deal with OOV words.
        const word = inputText[i];
        inputBuffer.set(this.wordIndex[word] + this.indexFrom, 0, i);
      }
      const input = inputBuffer.toTensor();
  
      console.log('Running inference');
      const beginMs = performance.now();
      const predictOut = this.model.predict(input);
      const score = predictOut.dataSync()[0];
      predictOut.dispose();
      const endMs = performance.now();
      console.log("score = " + score);
      console.log("Time elapsed = " + (endMs - beginMs));
      return {score: score, elapsed: (endMs - beginMs)};
    }
  };
