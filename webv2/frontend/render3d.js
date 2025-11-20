import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.min.js";

class Embedding3DRenderer {

	constructor(containerId) {
		this.container = document.getElementById(containerId);
	}

	_detectSOMData(data) {
		// SOM data has much smaller coordinate ranges (0-6) vs t-SNE (-50 to 50)
		const pos = data.sample_tsne_pos;
		const maxCoord = Math.max(Math.abs(pos[0]), Math.abs(pos[1]), Math.abs(pos[2]));
		return maxCoord < 10;  // If coordinates are small, it's SOM data
	}	

	render(data) {

		// -------------------------------------------------------------
		// 1) SCALING FACTOR — static for now, slider-ready
        // -------------------------------------------------------------
		const DEFAULT_TSNE_SCALE = 0.10;  
		const DEFAULT_SOM_SCALE = 3.0;

		// Detect which visualization mode based on data range
		const isSOM = this._detectSOMData(data);
		const SCALE = isSOM ? DEFAULT_SOM_SCALE : DEFAULT_TSNE_SCALE;		

		// -------------------------------------------------------------

		this.container.innerHTML = "";
		const width = this.container.clientWidth;
		const height = this.container.clientHeight;

		const scene = new THREE.Scene();
		const camera = new THREE.PerspectiveCamera(55, width / height, 0.01, 1000);
		camera.position.set(0, 0, 6);

		const renderer = new THREE.WebGLRenderer({ antialias: true });
		renderer.setSize(width, height);
		this.container.appendChild(renderer.domElement);

		renderer.setClearColor(0xffffff, 1);
		scene.background = new THREE.Color(0xffffff);

		const controls = new OrbitControls(camera, renderer.domElement);
		controls.enableDamping = true;
		controls.minDistance = 0.1;
		controls.maxDistance = 100;		
		controls.enableDamping = true;

		const top5Names = data.predictions.map(p => p.category);

		// Apply scaling for sample
		const samplePos = new THREE.Vector3(
			data.sample_tsne_pos[0] * SCALE,
			data.sample_tsne_pos[1] * SCALE,
			data.sample_tsne_pos[2] * SCALE
		);

		// --- Calculate bounding box of all category positions ---
		let minX = Infinity, maxX = -Infinity;
		let minY = Infinity, maxY = -Infinity;
		let minZ = Infinity, maxZ = -Infinity;

		Object.values(data.all_categories_tsne).forEach(pos => {
			const scaled = [pos[0] * SCALE, pos[1] * SCALE, pos[2] * SCALE];
			minX = Math.min(minX, scaled[0]);
			maxX = Math.max(maxX, scaled[0]);
			minY = Math.min(minY, scaled[1]);
			maxY = Math.max(maxY, scaled[1]);
			minZ = Math.min(minZ, scaled[2]);
			maxZ = Math.max(maxZ, scaled[2]);
		});

		const centerX = (minX + maxX) / 2;
		const centerY = (minY + maxY) / 2;
		const centerZ = (minZ + maxZ) / 2;
		const sizeX = maxX - minX;
		const sizeY = maxY - minY;
		const sizeZ = maxZ - minZ;
		const maxSize = Math.max(sizeX, sizeY, sizeZ);

		// Helper function to update camera for current container size
		const updateCameraForSize = () => {
			const w = this.container.clientWidth;
			const h = this.container.clientHeight;
			camera.aspect = w / h;
			camera.updateProjectionMatrix();
			renderer.setSize(w, h);

			// Recalculate camera position based on current container size
			const paddingFactor = isSOM ? 0.7 : 0.3;
			const padding = maxSize * paddingFactor;
			
			camera.position.set(
				centerX + padding,
				centerY + padding,
				centerZ + padding
			);
			controls.target.set(centerX, centerY, centerZ);
			controls.update();
		};

		// Initial setup
		updateCameraForSize();

		// auto-resize with responsive camera positioning
		window.addEventListener("resize", updateCameraForSize);       

		const meshes = {};

		const addSphere = (name, posArr, color) => {
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
			
			// Store mesh - handle overlapping positions
			if (!meshes[name]) {
				meshes[name] = mesh;
			}

			const label = this.createLabel(name);
			label.position.copy(mesh.position);
			label.position.y += 0.15;
			scene.add(label);
		};

		// Sample sphere - stored with special key to avoid collision
		const sampleGeo = new THREE.SphereGeometry(0.12, 20, 20);
		const sampleMat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
		const sampleMesh = new THREE.Mesh(sampleGeo, sampleMat);
		sampleMesh.position.copy(samplePos);
		scene.add(sampleMesh);
		meshes["__SAMPLE__"] = sampleMesh;

		const sampleLabel = this.createLabel("SAMPLE: " + data.actual_label);
		sampleLabel.position.copy(sampleMesh.position);
		sampleLabel.position.y += 0.15;
		scene.add(sampleLabel);

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
				meshes["__SAMPLE__"].position,
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
