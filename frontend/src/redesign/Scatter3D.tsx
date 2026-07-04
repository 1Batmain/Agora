import { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { ScatterRealPoint } from './scatterApi';
import type { SpatialTheme } from './contract';

/**
 * « Nuage 3D » — chaque point est une VRAIE contribution, projetée en UMAP 2D
 * (x, z) et surélevée en y par la densité locale (KDE JS léger).
 *
 * **Navigation** :
 * - Hover un point → les points du MÊME sous-thème grossissent (mise en évidence
 *   du groupe) + tooltip avec le texte de la contribution survolée.
 * - Double-clic sur un point → drill dans le sous-thème qui le contient.
 * - Fil d'Ariane (hors composant) pour remonter.
 */

const PLANE_SIZE = 10;
const HEIGHT_SCALE = 3.0;
const GRID_N = 48;
// Taille de base adaptative selon le nombre de points (plus il y en a, plus
// ils sont petits pour éviter l'overplot ; peu de points = gros points visibles).
const POINT_SIZE_SMALL = 0.14;  // >1000 points
const POINT_SIZE_MED = 0.22;    // 100-1000 points
const POINT_SIZE_LARGE = 0.35;  // <100 points
const POINT_HIGHLIGHT_MULT = 1.8;

function computeDensity(points: ScatterRealPoint[]) {
  if (!points.length) return { grid: new Float32Array(0), minX: 0, minZ: 0, cellW: 1, cellH: 1 };
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.z < minZ) minZ = p.z;
    if (p.z > maxZ) maxZ = p.z;
  }
  const spanX = (maxX - minX) || 1;
  const spanZ = (maxZ - minZ) || 1;
  const cellW = spanX / GRID_N;
  const cellH = spanZ / GRID_N;
  const grid = new Float32Array(GRID_N * GRID_N);
  for (const p of points) {
    const ix = Math.min(GRID_N - 1, Math.max(0, Math.floor((p.x - minX) / cellW)));
    const iz = Math.min(GRID_N - 1, Math.max(0, Math.floor((p.z - minZ) / cellH)));
    grid[iz * GRID_N + ix]++;
  }
  const smoothed = new Float32Array(GRID_N * GRID_N);
  for (let iz = 0; iz < GRID_N; iz++) {
    for (let ix = 0; ix < GRID_N; ix++) {
      let sum = 0, cnt = 0;
      for (let dz = -1; dz <= 1; dz++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = ix + dx, nz = iz + dz;
          if (nx >= 0 && nx < GRID_N && nz >= 0 && nz < GRID_N) {
            sum += grid[nz * GRID_N + nx];
            cnt++;
          }
        }
      }
      smoothed[iz * GRID_N + ix] = sum / cnt;
    }
  }
  let maxV = 0;
  for (let i = 0; i < smoothed.length; i++) if (smoothed[i] > maxV) maxV = smoothed[i];
  if (maxV > 0) for (let i = 0; i < smoothed.length; i++) smoothed[i] /= maxV;
  return { grid: smoothed, minX, minZ, cellW, cellH };
}

interface Scatter3DProps {
  points: ScatterRealPoint[];
  /** Map cluster_id (feuille) → ID du sous-thème enfant direct qui le contient. */
  clusterToThemeId?: Map<string, string>;
  /** Thèmes enfants directs du contexte courant (pour le titre du tooltip de groupe). */
  childThemes?: SpatialTheme[];
  /** Callback de drill (double-clic sur un point). */
  onDrill?: (themeId: string) => void;
}

export function Scatter3D({ points, clusterToThemeId, childThemes = [], onDrill }: Scatter3DProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<ScatterRealPoint | null>(null);
  const [hoveredThemeTitle, setHoveredThemeTitle] = useState<string | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null);

  // Index du point survolé (pour le raycaster dans le dblclick handler).
  const hoveredIdxRef = useRef<number>(-1);
  // Map cluster_id → themeId pour le drill.
  const clusterMapRef = useRef<Map<string, string>>(clusterToThemeId ?? new Map());
  clusterMapRef.current = clusterToThemeId ?? new Map();
  const onDrillRef = useRef<typeof onDrill>(onDrill);
  onDrillRef.current = onDrill;

  // Titre du thème pour le tooltip de groupe.
  const themeTitleMap = useMemo(() => {
    const m = new Map<string, string>();
    for (const t of childThemes) m.set(t.id, t.title || t.label);
    return m;
  }, [childThemes]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !points.length) return;

    // --- Bounds ---
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.z < minZ) minZ = p.z;
      if (p.z > maxZ) maxZ = p.z;
    }
    const spanX = (maxX - minX) || 1;
    const spanZ = (maxZ - minZ) || 1;

    const density = computeDensity(points);
    const sampleHeight = (x: number, z: number): number => {
      const ix = Math.min(GRID_N - 1, Math.max(0, Math.floor((x - density.minX) / density.cellW)));
      const iz = Math.min(GRID_N - 1, Math.max(0, Math.floor((z - density.minZ) / density.cellH)));
      return density.grid[iz * GRID_N + ix] || 0;
    };

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#f6f6f6');

    let width = mount.clientWidth || 1;
    let height = mount.clientHeight || 1;

    // --- Points ---
    const n = points.length;
    // Taille de point adaptative selon la densité.
    const baseSize = n > 1000 ? POINT_SIZE_SMALL : n > 100 ? POINT_SIZE_MED : POINT_SIZE_LARGE;
    // Caméra adaptative : plus proche quand peu de points (sinon tout tiny).
    const camDist = n > 1000 ? 14 : n > 100 ? 10 : 7;
    const camHeight = n > 1000 ? 8 : n > 100 ? 6 : 4.5;

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);
    camera.position.set(0, camHeight, camDist);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, camHeight * 0.15, 0);
    controls.minDistance = 3;
    controls.maxDistance = 40;

    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const tmpColor = new THREE.Color();

    // Map: themeId → Set d'indices de points (pour highlight de groupe).
    const themeToIndices = new Map<string, Set<number>>();
    const pointToTheme = new Map<number, string>();

    for (let i = 0; i < n; i++) {
      const p = points[i];
      const px = ((p.x - minX) / spanX - 0.5) * PLANE_SIZE;
      const pz = ((p.z - minZ) / spanZ - 0.5) * PLANE_SIZE;
      const py = sampleHeight(p.x, p.z) * HEIGHT_SCALE;
      positions[i * 3] = px;
      positions[i * 3 + 1] = py;
      positions[i * 3 + 2] = pz;

      if (p.color) tmpColor.set(p.color);
      else tmpColor.set('#000091');
      colors[i * 3] = tmpColor.r;
      colors[i * 3 + 1] = tmpColor.g;
      colors[i * 3 + 2] = tmpColor.b;
      sizes[i] = baseSize;

      // Associer le point à son thème (pour highlight + drill).
      const themeId = clusterMapRef.current.get(p.cluster_id ?? '');
      if (themeId) {
        pointToTheme.set(i, themeId);
        const set = themeToIndices.get(themeId) ?? new Set<number>();
        set.add(i);
        themeToIndices.set(themeId, set);
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    // Texture circulaire.
    const texCanvas = document.createElement('canvas');
    texCanvas.width = 64; texCanvas.height = 64;
    const tctx = texCanvas.getContext('2d')!;
    const grd = tctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.5, 'rgba(255,255,255,1)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    tctx.fillStyle = grd;
    tctx.fillRect(0, 0, 64, 64);
    const pointTexture = new THREE.CanvasTexture(texCanvas);

    // Shader Material avec taille variable par point (pour highlight).
    const pointMaterial = new THREE.ShaderMaterial({
      uniforms: {
        pointTexture: { value: pointTexture },
        uOpacity: { value: 0.85 },
      },
      vertexShader: `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (250.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform sampler2D pointTexture;
        uniform float uOpacity;
        varying vec3 vColor;
        void main() {
          vec4 tex = texture2D(pointTexture, gl_PointCoord);
          if (tex.a < 0.1) discard;
          gl_FragColor = vec4(vColor, tex.a * uOpacity);
        }
      `,
      transparent: true,
      depthWrite: false,
    });

    const pointCloud = new THREE.Points(geom, pointMaterial);
    scene.add(pointCloud);

    // Éclairage.
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // --- Raycaster ---
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: baseSize * 1.5 };
    const mouse = new THREE.Vector2();

    // État de highlight (thème survolé).
    let highlightedTheme: string | null = null;
    const sizeAttr = geom.attributes.size as THREE.BufferAttribute;
    const colorAttr = geom.attributes.color as THREE.BufferAttribute;
    // Sauvegarde des couleurs d'origine pour restaurer après dim.
    const origColors = new Float32Array(colors);

    const setHighlight = (themeId: string | null) => {
      if (themeId === highlightedTheme) return;
      highlightedTheme = themeId;

      if (themeId) {
        const highlightSet = themeToIndices.get(themeId);
        for (let i = 0; i < n; i++) {
          const inGroup = highlightSet?.has(i) ?? false;
          sizeAttr.array[i] = inGroup ? baseSize * POINT_HIGHLIGHT_MULT : baseSize * 0.6;
          // Dim les points hors groupe (assombrir vers gris).
          if (!inGroup) {
            colorAttr.array[i * 3] = origColors[i * 3] * 0.35;
            colorAttr.array[i * 3 + 1] = origColors[i * 3 + 1] * 0.35;
            colorAttr.array[i * 3 + 2] = origColors[i * 3 + 2] * 0.35;
          } else {
            colorAttr.array[i * 3] = origColors[i * 3];
            colorAttr.array[i * 3 + 1] = origColors[i * 3 + 1];
            colorAttr.array[i * 3 + 2] = origColors[i * 3 + 2];
          }
        }
      } else {
        // Restaurer tout.
        for (let i = 0; i < n; i++) {
          sizeAttr.array[i] = baseSize;
          colorAttr.array[i * 3] = origColors[i * 3];
          colorAttr.array[i * 3 + 1] = origColors[i * 3 + 1];
          colorAttr.array[i * 3 + 2] = origColors[i * 3 + 2];
        }
      }
      sizeAttr.needsUpdate = true;
      colorAttr.needsUpdate = true;
    };

    const onPointerMove = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(pointCloud, false);

      if (intersects.length > 0) {
        const idx = intersects[0].index ?? -1;
        if (idx >= 0 && idx < points.length) {
          const p = points[idx];
          hoveredIdxRef.current = idx;
          setHoveredPoint(p);

          // Highlight du groupe (sous-thème).
          const themeId = pointToTheme.get(idx);
          if (themeId) {
            setHighlight(themeId);
            setHoveredThemeTitle(themeTitleMap.get(themeId) ?? null);
          } else {
            setHighlight(null);
            setHoveredThemeTitle(null);
          }

          setHoverPos({ x: event.clientX - rect.left, y: event.clientY - rect.top });
          renderer.domElement.style.cursor = 'crosshair';
          return;
        }
      }
      hoveredIdxRef.current = -1;
      setHoveredPoint(null);
      setHoveredThemeTitle(null);
      setHighlight(null);
      renderer.domElement.style.cursor = 'grab';
    };

    const onDoubleClick = (event: MouseEvent) => {
      const idx = hoveredIdxRef.current;
      if (idx < 0) return;
      const p = points[idx];
      const themeId = clusterMapRef.current.get(p.cluster_id ?? '');
      if (themeId && onDrillRef.current) {
        onDrillRef.current(themeId);
      }
    };

    const onPointerLeave = () => {
      hoveredIdxRef.current = -1;
      setHoveredPoint(null);
      setHoveredThemeTitle(null);
      setHighlight(null);
    };

    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('dblclick', onDoubleClick);
    renderer.domElement.addEventListener('pointerleave', onPointerLeave);

    // --- Animation ---
    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // --- Resize ---
    const onResize = () => {
      width = mount.clientWidth || 1;
      height = mount.clientHeight || 1;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    // --- Cleanup ---
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('dblclick', onDoubleClick);
      renderer.domElement.removeEventListener('pointerleave', onPointerLeave);
      controls.dispose();
      geom.dispose();
      pointMaterial.dispose();
      pointTexture.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [points, clusterToThemeId, childThemes, themeTitleMap]);

  return (
    <div className="density3d">
      <div ref={mountRef} className="density3d__canvas" />
      {(hoveredPoint || hoveredThemeTitle) && hoverPos && (
        <div
          style={{
            position: 'absolute',
            left: Math.min(hoverPos.x + 14, (mountRef.current?.clientWidth || 400) - 320),
            top: Math.max(8, hoverPos.y - 80),
            maxWidth: 320,
            padding: '8px 12px',
            background: 'rgba(255,255,255,0.96)',
            border: '1px solid #000091',
            borderRadius: 6,
            fontSize: 13,
            lineHeight: 1.5,
            pointerEvents: 'none',
            zIndex: 10,
            boxShadow: '0 2px 12px rgba(0,0,0,0.18)',
          }}
        >
          {hoveredThemeTitle && (
            <div style={{ fontWeight: 700, marginBottom: 4, color: '#000091', fontSize: 13 }}>
              {hoveredThemeTitle}
            </div>
          )}
          {hoveredPoint?.text && (
            <div style={{ color: '#333', fontStyle: 'italic', marginBottom: 4 }}>
              « {hoveredPoint.text}{hoveredPoint.text.length >= 140 ? '…' : ''} »
            </div>
          )}
          <div style={{ color: '#666', fontSize: 11, borderTop: '1px solid #eee', paddingTop: 4 }}>
            double-clic pour explorer ce sous-thème
          </div>
        </div>
      )}
      <p className="density3d__legend">
        Nuage 3D · hover = mettre en évidence le sous-thème · double-clic = explorer · rotation =
        clic gauche, zoom = molette
      </p>
    </div>
  );
}
