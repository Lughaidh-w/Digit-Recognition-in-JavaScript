const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const getMnistData = require('./mnist_data.js');


const modelDir = './model';
if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir);
}


async function trainAndSaveModel() {
    const { trainingFeatures, trainingLabels, 
        testFeatures, testLabels } = getMnistData();

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [784]}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    
    model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ["accuracy"]});


    // Train the model
    await model.fit(trainingFeatures, trainingLabels, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });

    // Save the model
    await model.save(`file://${modelDir}`);
}


trainAndSaveModel();