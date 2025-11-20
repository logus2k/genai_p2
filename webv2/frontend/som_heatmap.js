import * as THREE from "three";

class SOMHeatmapRenderer {
	constructor(containerId) {
		this.container = document.getElementById(containerId);
	}

	render(data) {
		this.container.innerHTML = "";
		const width = this.container.clientWidth;
		const height = this.container.clientHeight;

		const scene = new THREE.Scene();
		scene.background = new THREE.Color(0xfafafa);

		const aspect = width / height;
		const frustumSize = 5;
		const camera = new THREE.OrthographicCamera(
			frustumSize * aspect / -2,
			frustumSize * aspect / 2,
			frustumSize / 2,
			frustumSize / -2,
			0.1, 100
		);
		camera.position.z = 5;
		let cameraZoom = frustumSize;

		const renderer = new THREE.WebGLRenderer({ antialias: true });
		renderer.setSize(width, height);
		this.container.appendChild(renderer.domElement);

		// Extract grid dimensions from data
		const gridX = 6, gridY = 6, gridZ = 6;

		// Build neuron occupancy map
		const neuronMap = new Map();
		const categoryPositions = data.all_categories_som || {};

		Object.entries(categoryPositions).forEach(([category, pos]) => {
			const key = `${pos[0]},${pos[1]},${pos[2]}`;
			if (!neuronMap.has(key)) {
				neuronMap.set(key, []);
			}
			neuronMap.get(key).push(category);
		});

		// Find max occupancy for normalization
		let maxOccupancy = 0;
		neuronMap.forEach(cats => {
			maxOccupancy = Math.max(maxOccupancy, cats.length);
		});

		// Colormap function (blue to red gradient)
		const getColorForDensity = (occupancy, max) => {
			const normalized = occupancy / max;
			// 0 = blue (0x4a90e2), 0.5 = yellow (0xf5d547), 1 = red (0xff4444)
			if (normalized < 0.5) {
				const t = normalized * 2;
				const r = Math.floor(0x4a + (0xf5 - 0x4a) * t);
				const g = Math.floor(0x90 + (0xd5 - 0x90) * t);
				const b = Math.floor(0xe2 + (0x47 - 0xe2) * t);
				return (r << 16) | (g << 8) | b;
			} else {
				const t = (normalized - 0.5) * 2;
				const r = Math.floor(0xf5 + (0xff - 0xf5) * t);
				const g = Math.floor(0xd5 + (0x44 - 0xd5) * t);
				const b = Math.floor(0x47 + (0x44 - 0x47) * t);
				return (r << 16) | (g << 8) | b;
			}
		};

		// Create 3D heatmap layers (Z slices)
		const layerSpacing = 0.15;
		for (let z = 0; z < gridZ; z++) {
			const layerGroup = new THREE.Group();

			for (let x = 0; x < gridX; x++) {
				for (let y = 0; y < gridY; y++) {
					const key = `${x},${y},${z}`;
					const occupancy = neuronMap.has(key) ? neuronMap.get(key).length : 0;
					const color = getColorForDensity(occupancy, maxOccupancy);

					// Box size based on occupancy
					const baseSize = 0.12;
					const sizeScale = occupancy > 0 ? 0.8 + 0.4 * (occupancy / maxOccupancy) : 0.6;
					const boxSize = baseSize * sizeScale;

					const geo = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
					const mat = new THREE.MeshPhongMaterial({
						color,
						emissive: color,
						emissiveIntensity: 0.3,
						wireframe: false
					});
					const mesh = new THREE.Mesh(geo, mat);

					// Position in grid (normalize to -3 to 3 range)
					mesh.position.x = (x - gridX / 2) * 0.4;
					mesh.position.y = (y - gridY / 2) * 0.4;
					mesh.position.z = (z - gridZ / 2) * layerSpacing;

					layerGroup.add(mesh);

					// Add text label for non-empty neurons
					if (occupancy > 0) {
						const label = this.createNumberLabel(occupancy.toString());
						label.position.copy(mesh.position);
						label.position.z += 0.15;
						layerGroup.add(label);
					}
				}
			}

			scene.add(layerGroup);
		}

		// Add sample position if available
		if (data.sample_som_pos) {
			const sampleGeo = new THREE.SphereGeometry(0.15, 16, 16);
			const sampleMat = new THREE.MeshPhongMaterial({
				color: 0xff4444,
				emissive: 0xff4444,
				emissiveIntensity: 0.8
			});
			const sampleMesh = new THREE.Mesh(sampleGeo, sampleMat);

			const [sx, sy, sz] = data.sample_som_pos;
			sampleMesh.position.set(
				(sx - gridX / 2) * 0.4,
				(sy - gridY / 2) * 0.4,
				(sz - gridZ / 2) * layerSpacing
			);
			scene.add(sampleMesh);

			// Sample label
			const sampleLabel = this.createLabel("SAMPLE");
			sampleLabel.position.copy(sampleMesh.position);
			sampleLabel.position.z += 0.25;
			scene.add(sampleLabel);
		}

		// Lighting
		const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
		light1.position.set(5, 5, 5);
		scene.add(light1);

		const light2 = new THREE.AmbientLight(0xffffff, 0.4);
		scene.add(light2);

		// Handle resize with camera repositioning
		const updateCameraForResize = () => {
			const w = this.container.clientWidth;
			const h = this.container.clientHeight;
			const aspect = w / h;
			camera.left = (cameraZoom * aspect) / -2;
			camera.right = (cameraZoom * aspect) / 2;
			camera.top = cameraZoom / 2;
			camera.bottom = cameraZoom / -2;
			camera.updateProjectionMatrix();
			renderer.setSize(w, h);
		};

		window.addEventListener("resize", updateCameraForResize);

		// Animation loop
		let rotationX = Math.PI / 4; // 45 degrees
		let rotationY = Math.PI / 4; // 45 degrees
		let isDragging = false;
		let previousMousePosition = { x: 0, y: 0 };

		renderer.domElement.addEventListener("mousedown", (e) => {
			isDragging = true;
			previousMousePosition = { x: e.clientX, y: e.clientY };
		});

		renderer.domElement.addEventListener("mousemove", (e) => {
			if (isDragging) {
				const deltaX = e.clientX - previousMousePosition.x;
				const deltaY = e.clientY - previousMousePosition.y;
				rotationY += deltaX * 0.005;
				rotationX += deltaY * 0.005;
				previousMousePosition = { x: e.clientX, y: e.clientY };
			}
		});

		renderer.domElement.addEventListener("mouseup", () => {
			isDragging = false;
		});

		renderer.domElement.addEventListener("wheel", (e) => {
			e.preventDefault();
			const zoomSpeed = 0.1;
			const zoomFactor = e.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed;
			cameraZoom *= zoomFactor;
			cameraZoom = Math.max(2, Math.min(20, cameraZoom)); // Clamp zoom

			const aspect = width / height;
			camera.left = (cameraZoom * aspect) / -2;
			camera.right = (cameraZoom * aspect) / 2;
			camera.top = cameraZoom / 2;
			camera.bottom = cameraZoom / -2;
			camera.updateProjectionMatrix();
		}, { passive: false });

		const animate = () => {
			requestAnimationFrame(animate);

			// Apply rotations to scene
			scene.rotation.x = rotationX;
			scene.rotation.y = rotationY;

			renderer.render(scene, camera);
		};

		animate();
	}

	createLabel(text) {
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");
		const fontSize = 32;

		ctx.font = `bold ${fontSize}px sans-serif`;
		const metrics = ctx.measureText(text);
		const padding = 10;

		canvas.width = metrics.width + padding * 2;
		canvas.height = fontSize + padding * 2;

		ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
		ctx.fillRect(0, 0, canvas.width, canvas.height);

		ctx.font = `bold ${fontSize}px sans-serif`;
		ctx.fillStyle = "#ffffff";
		ctx.textBaseline = "middle";
		ctx.fillText(text, padding, canvas.height / 2);

		const texture = new THREE.CanvasTexture(canvas);
		const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
		const sprite = new THREE.Sprite(material);

		sprite.scale.set(canvas.width / 200, canvas.height / 200, 1);
		return sprite;
	}

	createNumberLabel(text) {
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");
		const fontSize = 24;

		ctx.font = `${fontSize}px sans-serif`;
		const metrics = ctx.measureText(text);

		canvas.width = metrics.width + 8;
		canvas.height = fontSize + 8;

		ctx.fillStyle = "#000000";
		ctx.font = `${fontSize}px sans-serif`;
		ctx.textBaseline = "middle";
		ctx.fillText(text, 4, canvas.height / 2);

		const texture = new THREE.CanvasTexture(canvas);
		const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
		const sprite = new THREE.Sprite(material);

		sprite.scale.set(canvas.width / 150, canvas.height / 150, 1);
		return sprite;
	}
}

function renderSOMHeatmap(data) {
	const r = new SOMHeatmapRenderer("embeddingVisContainer");
	r.render(data);
}

window.renderSOMHeatmap = renderSOMHeatmap;
