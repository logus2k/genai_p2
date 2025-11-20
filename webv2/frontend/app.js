class FrontendApp {

	constructor() {
		this.socket = io("/", { path: "/scipredictor/socket.io/" });
		this.totalSamples = 0;
		this.visualizationMode = 'tsne'; // 'tsne', 'som', or 'som_heatmap'
		this.currentData = null;
		this._bindEvents();
		this._bindSocket();
	}

	_bindEvents() {
		document.getElementById("btnSearch").onclick = () => this.loadSample();
		document.getElementById("btnRandom").onclick = () => this.loadRandomSample();
		document.getElementById("btnVizTsne").onclick = () => this.setVisualization('tsne');
		document.getElementById("btnVizSom").onclick = () => this.setVisualization('som');
		document.getElementById("btnVizHeat").onclick = () => this.setVisualization('som_heatmap');
	}

	setVisualization(mode) {
		if (!this.currentData) return;

		// Check SOM data availability
		if ((mode === 'som' || mode === 'som_heatmap') && !this.currentData.sample_som_pos) {
			alert("SOM visualization not available. Please train the SOM model first.");
			return;
		}

		this.visualizationMode = mode;

		// Update button states
		document.getElementById("btnVizTsne").classList.toggle('active', mode === 'tsne');
		document.getElementById("btnVizSom").classList.toggle('active', mode === 'som');
		document.getElementById("btnVizHeat").classList.toggle('active', mode === 'som_heatmap');

		this._renderVisualization();
	}

	_renderVisualization() {
		if (!this.currentData) return;

		const vizData = {
			predictions: this.currentData.predictions,
			actual_label: this.currentData.actual_label,
			domain_colors: this.currentData.domain_colors
		};

		if (this.visualizationMode === 'tsne') {
			vizData.sample_tsne_pos = this.currentData.sample_tsne_pos;
			vizData.all_categories_tsne = this.currentData.all_categories_tsne;
			renderEmbeddingGraph(vizData);
		} else if (this.visualizationMode === 'som') {
			vizData.sample_som_pos = this.currentData.sample_som_pos;
			vizData.all_categories_som = this.currentData.all_categories_som;
			// Reuse embedding graph renderer with SOM data
			vizData.sample_tsne_pos = this.currentData.sample_som_pos;
			vizData.all_categories_tsne = this.currentData.all_categories_som;
			renderEmbeddingGraph(vizData);
		} else if (this.visualizationMode === 'som_heatmap') {
			vizData.sample_som_pos = this.currentData.sample_som_pos;
			vizData.all_categories_som = this.currentData.all_categories_som;
			renderSOMHeatmap(vizData);
		}
	}

	_bindSocket() {
		this.socket.on("connect", () => this.getDatasetInfo());

		this.socket.on("dataset_info", data => {
			this.totalSamples = data.total_samples;
			document.getElementById("datasetInfo").textContent =
				`Total samples: ${data.total_samples}`;

			this.loadRandomSample();
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
			this.currentData = data;
			this._renderVisualization();
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

		const actualLabel = data.actual_label;
		let matchFound = false;

		data.predictions.forEach((p, i) => {
			const div = document.createElement("div");
			div.className = "prediction-item";
			const isMatch = p.category === actualLabel;
			
			if (isMatch) {
				matchFound = true;
				div.className += " prediction-match";
			}

			div.innerHTML = `<strong>${i + 1}. [${p.domain}] ${p.category}</strong>:
				<span style="color:#007bff">${p.confidence.toFixed(4)}</span>
				${isMatch ? '<span style="margin-left:8px; color:#00aa00; font-weight:bold;">✓ MATCH</span>' : ''}`;
			list.appendChild(div);
		});

		// Add indicator if no match found in top 5
		if (!matchFound) {
			const noMatchDiv = document.createElement("div");
			noMatchDiv.style.cssText = "margin-top:10px; padding:8px; background:#ffe6e6; color:#cc0000; border-radius:4px; font-size:12px;";
			noMatchDiv.textContent = `Actual label "${actualLabel}" not in top 5 predictions`;
			list.appendChild(noMatchDiv);
		}

		document.getElementById("predictions").style.display = "block";
	}
}

window.onload = () => {
	window.app = new FrontendApp();
};
