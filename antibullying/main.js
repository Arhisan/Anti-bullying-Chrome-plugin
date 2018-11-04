const HOSTED_URLS = {
	model: 'https://downloader.disk.yandex.ru/disk/ec86989357711c59b50cf4c02fc7c998f8d94d4c50d58ffd8aee57c79ba65d9b/5bdd0ca3/GuzmeJG9vLceHGIrAYID3WRRxlUpT7a5_UfAZYVSkwztBEcz-V7U2kj9bRZiENSECX_v6HaXeLHEGBaMX2gYkw%3D%3D?uid=0&filename=model.json&disposition=attachment&hash=NZZr7/DxhMMeEzHdlKjDXtKJBmyGxEs0AzQl46auyiZwvbTkbRBNAalih01qq0MKq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=application%2Fjson&fsize=4461&hid=b0ccfaad98f373ab671b5fe674dc2146&media_type=unknown&tknv=v2',
	metadata: 'https://s95vla.storage.yandex.net/rdisk/3f82342bd7a38433f94fdf735757e4dfcfd0a2dc3f463c29efaa0ff296c1a962/5bdd0c77/GuzmeJG9vLceHGIrAYID3bYGdjR9y3IMSYFl-6wCZteT3lzYjnF-5ybAtoAz1K8gpVntoKl2BwnlKDgKMtLX8g==?uid=0&filename=metadata.json&disposition=attachment&hash=EsfvkCpd%2BDrp5cMsnyD%2BirCSIjQvQ%2BcylgR8THMRDcbeoy65e64d2Lw6rhOL0bEsq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=application%2Fjson&fsize=2456013&hid=d4fafc820ea3ba9bcb7b0fa27aa3d9bf&media_type=unknown&tknv=v2&rtoken=orRfiMvZUlsi&force_default=no&ycrid=na-3b4a5af1b67ed269fcde3a389557127c-downloader19f&ts=579b9af72cbc0&s=6407b6b3256c57f572ccb33a4949eefab74c572257f4da86d90e33f188f8a2e1&pb=U2FsdGVkX1-XmaqyVnLUOe23h65eEExgE-qnGUSpYXOkdxe8dHmjdko-0S7XmaN0wwBvo2R3xQ0UqI9uPt9Ya9SKEVnVax-nJuTmbHMPKa8'
};
  
const LOCAL_URLS = {
    model: chrome.extension.getURL('/resources/model.json'),
    metadata: chrome.extension.getURL('/resources/metadata.json')
};





/**
 * Test whether a given URL is retrievable.
 */
async function urlExists(url) {
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
async function loadHostedPretrainedModel(url) {
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
async function loadHostedMetadata(url) {
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

class SentimentPredictor {
    /**
     * Initializes the Sentiment demo.
     */
    async init(urls) {
		console.log("constructor start");
		this.urls = urls;
	
      	this.model = await loadHostedPretrainedModel(urls.model);
	  	await this.loadMetadata();
		console.log("constructor fin");
        //$('.im-mess--text.wall_module._im_log_body:contains("лол")').css('background-color', 'red');
        var messages = $('.TweetTextSize.js-tweet-text.tweet-text');
        //messages.each(function() {
        //    $(this).html(pred($(this).text()).score);
        //});


       // window.setInterval(function(a) { 
        //    return function(){
                var th = 0.3;
                for (let i = 0; i < messages.length; i++){
                    var score = this.pred($(messages[i]).text()).score;
                    var inject;
                    if(score > th){
                        inject = '<div  style="filter:url(#wherearemyglasses); filter:  blur(2px); -webkit-filter: blur(3px);">' 
                        +$(messages[i]).text()
                        +"\n"+score+'</div>';
                    }else{
                        inject = $(messages[i]).text() ;//+"\n"+score;
                    }

                    $(messages[i]).html(inject);
                   // $(messages[i]).css('color', getRandomColor())
                }
        //    }
       // }(this),4000)

      	return this;
    }
  
    async loadMetadata() {
      const sentimentMetadata = await loadHostedMetadata(this.urls.metadata);
      this.indexFrom = 0//sentimentMetadata['index_from'];
      this.maxLen = sentimentMetadata['max_len'];
      console.log('indexFrom = ' + this.indexFrom);
      console.log('maxLen = ' + this.maxLen);
  
      this.wordIndex = sentimentMetadata['word_index']
    }
  
    pred(text) {
        /// Convert to lower case and remove all punctuations. !"#$%&()*+,-./:;<=>?@[\]^_`{|}~        .,\/#!$%\^&\*;:{}=\-_`~()
        const inputText = text.trim().toLowerCase().replace(/["#$%&()*+,-./:;<=>?@\[\]^_`{|}~]/g,"").replace(/\s{2,}/g," ").split(' ') // replace(/(\.|\,|\!)/g, '').split(' ');
        // Look up word indices.
        const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
        if (this.maxLen > inputText.length)
          for (let i = 0; i < inputText.length; ++i) {
            // TODO(cais): Deal with OOV words.
            const word = inputText[i];
            inputBuffer.set(this.wordIndex[word] + this.indexFrom, 0, i + this.maxLen - inputText.length);
          }
        else
          for (let i = 0; i < inputText.length; ++i) {
            // TODO(cais): Deal with OOV words.
            const word = inputText[i];
            inputBuffer.set(this.wordIndex[word] + this.indexFrom, 0, i);
          }
        // for (let i = inputText.length - 1; i >= 0; --i) {
        //   // TODO(cais): Deal with OOV words.
        //   const word = inputText[i];
        //   inputBuffer.set(this.wordIndex[word] + this.indexFrom, 0, i);
        // }
        const input = inputBuffer.toTensor();
  
        //console.log("INPUT TENSOR");
        console.log(inputBuffer);
  
        const beginMs = performance.now();
        const predictOut = this.model.predict(input);
        const score = predictOut.dataSync()[1];
        predictOut.dispose();
        const endMs = performance.now();
        console.log(score+" "+ (endMs-beginMs));
        return {score: score, elapsed: (endMs - beginMs)};
      }
};



const checker = new SentimentPredictor().init(LOCAL_URLS); 
$(document).ready(function() 
{
    var s = document.createElement("script");
    s.type = "text/javascript";
    s.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js";
    // Use any selector
    $("head").append(s);
});

//$('.im-mess--text.wall_module._im_log_body:contains("лол")').css('background-color', 'red');


$(document).ready(function() {
    //MyFunc("hello");
});

function MyFunc(text) {
    alert(text);
}

function Hash(s)
{
	return s.split("").reduce(
		function(a,b)
		{
			a = ((a<<5)-a)+b.charCodeAt(0);
			return a&a;
		},0);              
}

function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  } 
