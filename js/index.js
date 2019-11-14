let eixoX = [];
let eixoY = [];

let entradaX = 0;
let entradaY = 0;

let model = null;
let history = null;

let spanErro = document.getElementById("erro");
let spanStatus = document.getElementById("status");

window.addEventListener('load', () => {
	var ajax = new XMLHttpRequest();

	ajax.open("GET", "dados.csv", true);

	ajax.send();

	ajax.onreadystatechange = () => {
		if (ajax.readyState == 4 && ajax.status == 200) {
			var data = ajax.responseText;

			eixoX = [];
			eixoY = [];

			let lines = data.split(/\r?\n|\r/);
			for (let i = 1; i < lines.length; i++) {
				let celulas = lines[i].split(",");
				eixoX.push([Number(celulas[0]), Number(celulas[1]), Number(celulas[2])]);
				eixoY.push([Number(celulas[3])]);
			}

		}
	}
});

let btnDados = document.getElementById("btnHistory");

document.getElementById("btnTreinar").addEventListener("click", async () => {
	spanStatus.innerText = "Em Treinamento";
	const taxaErro = parseFloat(document.getElementById("taxa").value);
	const nIteracoes = parseInt(document.getElementById("iteracoes").value);
	await treinamento(taxaErro, nIteracoes);
	btnDados.classList.remove("invisible");
	btnDados.classList.add("visible");
	spanStatus.innerText = "Treinado";
});

btnDados.addEventListener("click", async () => {
	const surface = { name: 'Taxa de Erro', tab: 'Taxa de Erro' };
	tfvis.show.history(surface, history, ['loss']);
	tfvis.show.modelSummary({ name: 'Taxa de Erro', tab: 'Dados do Modelo' }, model);
});

async function treinamento(taxaAceita, nIteracoes) {

	const x = tf.tensor(eixoX);
	const y = tf.tensor(eixoY);

	let taxa = 1;
	taxaAceita = taxaAceita / 100;

	//while (taxa > taxaAceita) {

		model = tf.sequential();

		let inputLayer = tf.layers.dense({ units: 3, inputShape: [3], activation: 'tanh' });
		let hiddenLayer1 = tf.layers.dense({ units: 3, inputShape: [3], activation: 'sigmoid' });
		let hiddenLayer2 = tf.layers.dense({ units: 2, inputShape: [3], activation: 'sigmoid' });
		let outputLayer = tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' });

		model.add(inputLayer);
		model.add(hiddenLayer1);
		model.add(hiddenLayer2);
		model.add(outputLayer);

		model.compile({
			loss: tf.losses.meanSquaredError,
			optimizer: tf.train.rmsprop(.05)
		});

		history = await model.fit(x, y, {
			epochs: nIteracoes,
			callbacks: {
				onEpochEnd: (epoch, log) => {

					taxa = parseFloat(log.loss).toFixed(4);

					if (epoch % 10 == 0) {
						spanErro.innerText = (taxa * 100).toFixed(4);
					}

					if (taxa < taxaAceita) {
						model.stopTraining = true;
						spanErro.innerText = (taxa * 100).toFixed(4);
					}
				}
			}
		});
	//}

}

function predicao(entradaI) {
	const z = tf.tensor([[entradaI, entradaX, entradaY]]);
	let output = model.predict(z).round().arraySync();
	return output;
}

document.getElementById("btnExecutar").addEventListener("click", function () {
	entradaX = Number(document.getElementById("entradaX").value);
	entradaY = Number(document.getElementById("entradaY").value);

	for (let i = 1; i <= 5; ++i) {
		document.getElementById("tv" + i).innerText = predicao(i);
	}
});
