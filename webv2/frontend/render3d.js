// render3d.js

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.min.js";

class Embedding3DRenderer {

	constructor(containerId) {
		this.container = document.getElementById(containerId);
	}

	render(data) {
		this.container.innerHTML = "";
		const width = this.container.clientWidth;
		const height = this.container.clientHeight;

		const scene = new THREE.Scene();
		const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
		camera.position.set(0, 0, 6);

		const renderer = new THREE.WebGLRenderer({ antialias: true });
		renderer.setSize(width, height);
		this.container.appendChild(renderer.domElement);

		renderer.setClearColor(0xffffff, 1);
		scene.background = new THREE.Color(0xffffff);

		const controls = new OrbitControls(camera, renderer.domElement);

		const normalize = v => {
			const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
			return v.map(x => x / (n + 1e-9));
		};

		const cosineDistance = (a, b) => {
			let dot = 0;
			for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
			return 1 - dot;
		};

		const sampleVec = normalize(data.sample_embedding);

		const allCats = Object.keys(data.category_embeddings).map(c => ({
			name: c,
			vector: normalize(data.category_embeddings[c])
		}));

		const top5Names = data.predictions.map(p => p.category);

		const nodes = [];

		// sample (center)
		nodes.push({
			name: "__sample__",
			label: data.actual_label,
			vector: sampleVec,
			type: "sample",
			pos: new THREE.Vector3(0, 0, 0)
		});

		// all categories (top-5 + rest)
		allCats.forEach(c => {
			nodes.push({
				name: c.name,
				vector: c.vector,
				type: top5Names.includes(c.name) ? "top5" : "other",
				pos: new THREE.Vector3(
					Math.random() * 2 - 1,
					Math.random() * 2 - 1,
					Math.random() * 2 - 1
				)
			});
		});

		const N = nodes.length;

		// layout iterations
		for (let iter = 0; iter < 150; iter++) {
			for (let i = 1; i < N; i++) {
				const anchor = nodes[0].pos;
				const node = nodes[i];
				const d = cosineDistance(nodes[0].vector, node.vector);
				const desired = 0.5 + d * 3.0;

				const delta = node.pos.clone().sub(anchor);
				const dist = delta.length() + 1e-9;
				const force = (dist - desired) * 0.03;

				delta.normalize().multiplyScalar(force);
				node.pos.sub(delta);
			}
		}

		const sphereMeshes = {};
		const top5Lines = [];

		nodes.forEach(node => {
			let geo, mat, mesh;

			if (node.type === "sample") {
				geo = new THREE.SphereGeometry(0.18, 32, 32);
				mat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
			} else if (node.type === "top5") {
				geo = new THREE.SphereGeometry(0.12, 20, 20);
				mat = new THREE.MeshBasicMaterial({ color: 0x0088ff });
			} else {
				geo = new THREE.SphereGeometry(0.12, 20, 20);
				mat = new THREE.MeshBasicMaterial({ color: 0x00aa00 });
			}

			mesh = new THREE.Mesh(geo, mat);
			mesh.position.copy(node.pos);
			scene.add(mesh);

			const labelText =
				node.type === "sample"
					? this.shortenName(node.label)
					: this.shortenName(node.name);

			const label = this.createBillboardLabel(labelText);
			label.position.copy(mesh.position);
			label.position.y += 0.15;
			scene.add(label);

			sphereMeshes[node.name] = mesh;
		});

		// Connect sample → each top-5
		const sampleMesh = sphereMeshes["__sample__"];

		nodes.forEach(node => {
			if (node.type !== "top5") return;

			const endMesh = sphereMeshes[node.name];

			const pts = [];
			pts.push(sampleMesh.position.clone());
			pts.push(endMesh.position.clone());

			const geom = new THREE.BufferGeometry().setFromPoints(pts);

			const line = new THREE.Line(
				geom,
				new THREE.LineBasicMaterial({
					color: 0x000000,
					linewidth: 1.4
				})
			);

			scene.add(line);
			top5Lines.push(line);
		});

		// FIX: Orbit pivot centered on sample sphere
		controls.target.copy(nodes[0].pos);
		controls.update();

		const animate = () => {
			requestAnimationFrame(animate);
			controls.update();
			renderer.render(scene, camera);
		};

		animate();
	}

	shortenName(fullName) {
		if (typeof fullName !== "string") return "";
		const match = fullName.match(/\(([^)]+)\)\s*$/);
		return match ? match[1] : fullName;
	}

	createBillboardLabel(text) {
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");

		const fontSize = 36;
		ctx.font = fontSize + "px sans-serif";

		const padding = 20;
		const textWidth = ctx.measureText(text).width;

		canvas.width = textWidth + padding;
		canvas.height = fontSize + padding;

		ctx.font = fontSize + "px sans-serif";
		ctx.fillStyle = "#000000";
		ctx.textBaseline = "middle";
		ctx.fillText(text, padding / 2, canvas.height / 2);

		const texture = new THREE.CanvasTexture(canvas);
		const material = new THREE.SpriteMaterial({
			map: texture,
			transparent: true
		});
		const sprite = new THREE.Sprite(material);

		sprite.scale.set(
			canvas.width / 300,
			canvas.height / 300,
			1
		);

		return sprite;
	}
}

function renderEmbeddingGraph(data) {
	const r = new Embedding3DRenderer("embeddingVisContainer");
	r.render(data);
}

window.renderEmbeddingGraph = renderEmbeddingGraph;
