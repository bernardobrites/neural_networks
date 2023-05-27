//nomedoficheiro3.js
// Executar a 1a rede neutral - Ensinar a lógica dos operadores and, num fic

const brain = require("brain.js");
const net = new brain.NeuralNetwork();
net.train([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [0] },
  { input: [1, 0], output: [0] },
  { input: [1, 1], output: [1] },
]);
const output00 = parseFloat(net.run([0, 0])).toFixed(0);
const output01 = parseFloat(net.run([0, 1])).toFixed(0);
const output10 = parseFloat(net.run([1, 0])).toFixed(0);
const output11 = parseFloat(net.run([1, 1])).toFixed(0);
console.log("0 and 0: ${output00}");
console.log("0 and 1: ${output01}");
console.log("1 and 0: ${output10}");
console.log("1 and 1: ${output11}");

// Guardem e testem agora no terminal onde nomedoficheiro3
