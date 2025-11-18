function renderEmbeddingGraph(data) {
    const container = document.getElementById("embeddingVisContainer");
    container.innerHTML = "";

    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, width/height, 0.1, 1000);
    camera.position.set(0, 0, 6);

    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const sampleVector = data.sample_embedding;
    const predEmbeds = data.prediction_embeddings;

    // Normalize vectors
    function normalize(v) {
        const n = Math.sqrt(v.reduce((s,x)=>s+x*x,0));
        return v.map(x => x / (n+1e-9));
    }

    const sampleNorm = normalize(sampleVector);

    const preds = data.predictions.map(p => ({
        name: p.category,
        vector: normalize(predEmbeds[p.category])
    }));

    function cosineDistance(a, b) {
        let dot = 0;
        for (let i=0;i<a.length;i++) dot += a[i]*b[i];
        return 1 - dot;
    }

    // Build nodes
    const nodes = [];

    // sample node
    nodes.push({
        name: "sample",
        vector: sampleNorm,
        pos: new THREE.Vector3(0,0,0)
    });

    // predicted nodes, random init
    preds.forEach(p => nodes.push({
        name: p.name,
        vector: p.vector,
        pos: new THREE.Vector3(
            Math.random()*2-1,
            Math.random()*2-1,
            Math.random()*2-1
        )
    }));

    // Force layout (sample is anchor)
    const N = nodes.length;
    for (let iter = 0; iter < 150; iter++) {
        for (let i = 1; i < N; i++) {
            const samplePos = nodes[0].pos;
            const node = nodes[i];

            const d = cosineDistance(nodes[0].vector, node.vector);
            const desired = 0.5 + d * 3.0;

            const delta = node.pos.clone().sub(samplePos);
            const dist = delta.length() + 1e-9;
            const force = (dist - desired) * 0.03;

            delta.normalize().multiplyScalar(force);
            node.pos.sub(delta);
        }
    }

    // Render spheres
    nodes.forEach((node, idx) => {
        const geo = new THREE.SphereGeometry(idx === 0 ? 0.18 : 0.12, 20, 20);
        const mat = new THREE.MeshBasicMaterial({ color: idx === 0 ? 0xff4444 : 0x0088ff });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(node.pos);
        scene.add(mesh);
    });

    // Animation
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}
