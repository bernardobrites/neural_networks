const brain = require("brain.js");

// Primeiro treinamento com configurações originais
const config1 = {
  binaryThresh: 1,
  hiddenLayers: [3],
  activation: "sigmoid",
  leakyReluAlpha: 0.01,
  inputSize: 2,
  inputRange: 2,
  outputSize: 1,
  learningRate: 0.01,
  decayRate: 0.999,
};
const net1 = new brain.NeuralNetwork(config1);
net1.train([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
]);
const output1 = net1.run([0, 1]);
console.log("Primeiro treinamento:", output1);

// Segundo treinamento com configurações alteradas
const config2 = {
  binaryThresh: 1,
  hiddenLayers: [1],
  activation: "sigmoid",
  leakyReluAlpha: 0.01,
  inputSize: 2,
  inputRange: 2,
  outputSize: 1,
  learningRate: 0.01,
  decayRate: 0.999,
};
const net2 = new brain.NeuralNetwork(config2);
net2.train([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
]);
const output2 = net2.run([0, 1]);
console.log("Segundo treinamento:", output2);
