// app.js


class FrontendApp {

	constructor() {
		this.socket = io("/", { path: "/scipredictor/socket.io/" });
		this.totalSamples = 0;
		this._bindEvents();
		this._bindSocket();
	}

	_bindEvents() {
		document.getElementById("btnSearch").onclick = () => this.loadSample();
		document.getElementById("btnRandom").onclick = () => this.loadRandomSample();
	}

	_bindSocket() {
		this.socket.on("connect", () => this.getDatasetInfo());

		this.socket.on("dataset_info", data => {
			this.totalSamples = data.total_samples;
			document.getElementById("datasetInfo").textContent =
				// `Total samples: ${data.total_samples} | Columns: ${data.columns.join(", ")}`;
				`Total samples: ${data.total_samples}`;
		});

		this.socket.on("sample", data => {
			this._displaySample(data);
			document.getElementById("loading").style.display = "none";
			document.getElementById("error").style.display = "none";
			this.socket.emit("predict_sample", { index: data.index });

			if (data.pdf_url) {
				const iframe = document.getElementById("pdfViewer");
				iframe.src = data.pdf_url + "#zoom=page-fit";
				iframe.style.display = "block";
			}
		});

        this.socket.on("prediction_result", data => {
            this._displayPredictions(data);
        });

		this.socket.on("tsne_result", data => {
			this._displayPredictions(data);
			renderEmbeddingGraph(data);
		});

		this.socket.on("error", data => {
			document.getElementById("error").textContent = data.message;
			document.getElementById("error").style.display = "block";
		});
	}

	loadSample() {
		const index = parseInt(document.getElementById("sampleIndex").value);
		if (isNaN(index) || index < 0) {
			document.getElementById("error").textContent = "Invalid index";
			document.getElementById("error").style.display = "block";
			return;
		}
		document.getElementById("loading").style.display = "block";
		this.socket.emit("get_sample", { index });
	}

	loadRandomSample() {
		const idx = Math.floor(Math.random() * this.totalSamples);
		document.getElementById("sampleIndex").value = idx;
		this.loadSample();
	}

	getDatasetInfo() {
		this.socket.emit("get_dataset_info", {});
	}

	_displaySample(data) {
		document.getElementById("sampleIndexDisplay").textContent = data.index;
		document.getElementById("actualLabel").textContent = data.actual_label;
		document.getElementById("textPreview").textContent = data.text.substring(0, 500) + "...";
		document.getElementById("sampleInfo").style.display = "block";
	}

	_displayPredictions(data) {
		const list = document.getElementById("predictionList");
		list.innerHTML = "";

		data.predictions.forEach((p, i) => {
			const div = document.createElement("div");
			div.className = "prediction-item";
			div.innerHTML = `<strong>${i + 1}. [${p.domain}] ${p.category}</strong>:
				<span style="color:#007bff">${p.confidence.toFixed(4)}</span>`;
			list.appendChild(div);
		});

		document.getElementById("predictions").style.display = "block";
	}
}

window.onload = () => {
	window.app = new FrontendApp();
    app.loadSample();
};
