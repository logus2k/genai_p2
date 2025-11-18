import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.min.js";

class Embedding3DRenderer {

	constructor(containerId) {
		this.container = document.getElementById(containerId);
	}

	render(data) {

		// -------------------------------------------------------------
		// 1) SCALING FACTOR — static for now, slider-ready
        // -------------------------------------------------------------
		const DEFAULT_TSNE_SCALE = 0.10;  
		const SCALE = DEFAULT_TSNE_SCALE;

		// -------------------------------------------------------------

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
		controls.enableDamping = true;

		const top5Names = data.predictions.map(p => p.category);

		// auto-resize
		window.addEventListener("resize", () => {
			const width = this.container.clientWidth;
			const height = this.container.clientHeight;
			camera.aspect = width / height;
			camera.updateProjectionMatrix();
			renderer.setSize(width, height);
		});

		// -------------------------------------------------------------
		// 2) Apply scaling for sample
		// -------------------------------------------------------------
		const samplePos = new THREE.Vector3(
			data.sample_tsne_pos[0] * SCALE,
			data.sample_tsne_pos[1] * SCALE,
			data.sample_tsne_pos[2] * SCALE
		);

        // --- Center camera on the sample sphere ---
        controls.target.copy(samplePos);

        // Move camera backward along Z so the sample sphere is in front
        const distance = samplePos.length() * 0.2 + 4;
        camera.position.copy(samplePos.clone().add(new THREE.Vector3(0, 0, distance)));

        // Apply updated control target
        controls.update();        

		const meshes = {};

		const addSphere = (name, posArr, color) => {
			// -------------------------------------------------------------
			// 3) Scale positions for all spheres
			// -------------------------------------------------------------
			const scaled = [
				posArr[0] * SCALE,
				posArr[1] * SCALE,
				posArr[2] * SCALE
			];

			const geo = new THREE.SphereGeometry(0.12, 20, 20);
			const mat = new THREE.MeshBasicMaterial({ color });
			const mesh = new THREE.Mesh(geo, mat);

			mesh.position.copy(new THREE.Vector3(...scaled));
			scene.add(mesh);
			meshes[name] = mesh;

			const label = this.createLabel(name);
			label.position.copy(mesh.position);
			label.position.y += 0.15;
			scene.add(label);
		};

		// Sample sphere
		addSphere(data.actual_label, data.sample_tsne_pos, 0xff4444);

        // Category spheres
		Object.entries(data.all_categories_tsne).forEach(([cat, pos]) => {
			const isTop5 = top5Names.includes(cat);
			const dom = cat.includes("(")
				? cat.split("(")[0].trim().split(" ")[0]
				: "other";

			const color = isTop5
				? 0x0088ff
				: (data.domain_colors[dom] || 0x00aa00);

			addSphere(cat, pos, color);
		});

		// Lines to top-5
		top5Names.forEach(name => {
			if (!meshes[name]) return;

			const lineGeo = new THREE.BufferGeometry().setFromPoints([
				meshes[data.actual_label].position,
				meshes[name].position
			]);

			const lineMat = new THREE.LineBasicMaterial({ color: 0x000000 });
			const line = new THREE.Line(lineGeo, lineMat);
			scene.add(line);
		});

		const animate = () => {
			requestAnimationFrame(animate);
			controls.update();
			renderer.render(scene, camera);
		};

		animate();
	}

	createLabel(text) {
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");

		const fontSize = 40;
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
