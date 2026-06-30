import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { DensityPayload } from './densityApi';

/**
 * « Paysage de densité 3D » — une SURFACE, pas un nuage de points. Le plan (x, z)
 * est la projection UMAP 2D des embeddings PRÉ-clustering ; la hauteur y est la
 * densité locale (KDE) sur une grille nx×nz. Pics = zones denses (futurs thèmes),
 * vallées = clairsemé.
 *
 * Composant de RENDU PUR : il reçoit le `payload` de densité PRÉCALCULÉ (chargé en
 * amont depuis le cache `GET /density`, jamais recalculé à la requête) et le dessine.
 * Rendu three.js : une `PlaneGeometry(nx-1, nz-1)` dont chaque sommet prend
 * `y = heights[iz][ix]` (normalisé par `zmax`), coloré par un dégradé de hauteur
 * (rampe Bleu France clair→foncé), éclairage doux, `OrbitControls` (rotation/zoom
 * souris). Fond clair épuré. Resize géré, nettoyage three au démontage.
 */

// Rampe Bleu France : vallées en bleu pâle (#ececfe) → pics en bleu profond (#000091).
const LOW = new THREE.Color('#ececfe');
const HIGH = new THREE.Color('#000091');

// Étendue du terrain dans la scène (unités three) et amplitude verticale des pics.
const PLANE_SIZE = 10;
const HEIGHT_SCALE = 3.2;

export function Density3D({ payload }: { payload: DensityPayload }) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  // Construit/anime la scène three quand la grille (précalculée) est prête.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !payload) return;

    const { nx, nz, heights, zmax } = payload;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#f6f6f6');

    let width = mount.clientWidth || 1;
    let height = mount.clientHeight || 1;

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(0, 9, 12);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    // Surface : PlaneGeometry (nx-1)×(nz-1) segments → nx×nz sommets. On déplace
    // chaque sommet en y selon la densité normalisée et on lui attribue une couleur
    // de la rampe Bleu France (vertex colors → dégradé continu par hauteur).
    const geom = new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE, nx - 1, nz - 1);
    geom.rotateX(-Math.PI / 2); // plan XY → plan XZ (sol horizontal, hauteur en Y).
    const pos = geom.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(pos.count * 3);
    const norm = zmax > 0 ? zmax : 1;
    const c = new THREE.Color();
    for (let iz = 0; iz < nz; iz++) {
      for (let ix = 0; ix < nx; ix++) {
        const i = iz * nx + ix;
        const h = (heights[iz][ix] ?? 0) / norm; // 0..1
        pos.setY(i, h * HEIGHT_SCALE);
        c.copy(LOW).lerp(HIGH, h);
        colors[i * 3] = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
      }
    }
    pos.needsUpdate = true;
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.0,
      side: THREE.DoubleSide,
      flatShading: false,
    });
    const mesh = new THREE.Mesh(geom, material);
    scene.add(mesh);

    // Éclairage simple : ambiance douce + une directionnelle pour le relief.
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(6, 12, 8);
    scene.add(key);

    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize : suit la taille du conteneur (ResizeObserver + repli window).
    const onResize = () => {
      width = mount.clientWidth || 1;
      height = mount.clientHeight || 1;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);
    window.addEventListener('resize', onResize);

    // --- Cleanup : libère three proprement au démontage / changement de grille. ---
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', onResize);
      controls.dispose();
      geom.dispose();
      material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };
  }, [payload]);

  return (
    <div className="density3d">
      <div ref={mountRef} className="density3d__canvas" />
      <p className="density3d__legend">
        Surface = projection UMAP 2D des contributions · hauteur = densité (pics = thèmes
        denses)
      </p>
    </div>
  );
}
