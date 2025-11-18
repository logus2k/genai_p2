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

		renderer.setClearColor(0xFFFFFF, 1);
		scene.background = new THREE.Color(0xFFFFFF);

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
		const preds = data.predictions.map(p => ({
			name: p.category,
			vector: normalize(data.prediction_embeddings[p.category])
		}));

		const nodes = [];

        nodes.push({
            name: data.actual_label,
            vector: sampleVec,
            pos: new THREE.Vector3(0, 0, 0)
        });

		preds.forEach(p =>
			nodes.push({
				name: p.name,
				vector: p.vector,
				pos: new THREE.Vector3(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1)
			})
		);

		const N = nodes.length;

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

		nodes.forEach((node, idx) => {
			let geo, mat, mesh;

			if (idx === 0) {
				geo = new THREE.SphereGeometry(0.18, 32, 32);
				mat = new THREE.MeshBasicMaterial({ color: 0xf33838 });
			} else {
				geo = new THREE.SphereGeometry(0.12, 20, 20);
				mat = new THREE.MeshBasicMaterial({ color: 0x0088ff });
			}

			mesh = new THREE.Mesh(geo, mat);
			mesh.position.copy(node.pos);
			scene.add(mesh);

			if (idx === 0) {
				const billboard = this.createBillboardLabel(this.shortenName(node.name));
				billboard.position.set(0, 0.22, 0);
				mesh.add(billboard);
			}

			if (idx !== 0) {
				const label = this.createLabel(this.shortenName(node.name));
				label.position.copy(mesh.position);
				label.position.y += 0.15;
				scene.add(label);
			}
		});

		const animate = () => {
			requestAnimationFrame(animate);
			controls.update();
			renderer.render(scene, camera);
		};

		animate();
	}

	shortenName(fullName) {
		const match = fullName.match(/\(([^)]+)\)\s*$/);
		return match ? match[1] : fullName;
	}

	createLabel(text) {
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');

		const fontSize = 35;
		const scaleFactor = 300;

		ctx.font = fontSize + 'px sans-serif';

		const padding = 20;
		const textWidth = ctx.measureText(text).width;

		canvas.width = textWidth + padding;
		canvas.height = fontSize + padding;

		ctx.font = fontSize + 'px sans-serif';
		ctx.fillStyle = '#000000';
		ctx.textBaseline = 'middle';
		ctx.textAlign = 'left';
		ctx.fillText(text, padding / 2, canvas.height / 2);

		const texture = new THREE.CanvasTexture(canvas);
		const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
		const sprite = new THREE.Sprite(material);

		sprite.scale.set(canvas.width / scaleFactor, canvas.height / scaleFactor, 1);

		return sprite;
	}

	createBillboardLabel(text) {
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');

		const fontSize = 48;
		const padding = 20;
		ctx.font = fontSize + 'px sans-serif';

		const textWidth = ctx.measureText(text).width;

		canvas.width = textWidth + padding;
		canvas.height = fontSize + padding;

		ctx.font = fontSize + 'px sans-serif';
		ctx.fillStyle = '#000000';
		ctx.textBaseline = 'middle';
		ctx.textAlign = 'center';
		ctx.fillText(text, canvas.width / 2, canvas.height / 2);

		const texture = new THREE.CanvasTexture(canvas);
		const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
		const sprite = new THREE.Sprite(material);

		const scaleFactor = 350;
		sprite.scale.set(canvas.width / scaleFactor, canvas.height / scaleFactor, 1);

		return sprite;
	}
}

function renderEmbeddingGraph(data) {
	const renderer = new Embedding3DRenderer("embeddingVisContainer");
	renderer.render(data);
}

window.renderEmbeddingGraph = renderEmbeddingGraph;
