import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { buildBrickScene, stepOrder } from "../three/bricks";
import type { Build } from "../types";

function Model({ build }: { build: Build }) {
  const material = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.32, metalness: 0 }), []);
  const scene = useMemo(() => {
    const s = buildBrickScene(build, stepOrder(build), material);
    s.setVisible(build.bricks.length);
    s.group.traverse((o) => {
      o.castShadow = true;
      o.receiveShadow = true;
    });
    return s;
  }, [build, material]);
  useEffect(() => () => scene.dispose(), [scene]);
  return <primitive object={scene.group} />;
}

export default function BrickViewer({ build }: { build: Build }) {
  const span = Math.max(build.dims.x, build.dims.y, build.dims.z * 1.2);
  const target: [number, number, number] = [0, (build.dims.z * 1.2) / 2, 0];
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [-span * 1.1, span * 1.0, span * 1.4], fov: 35, near: 0.1, far: span * 20 }}
      aria-label="Rotatable 3D model of the build. Drag to rotate, scroll to zoom."
    >
      <hemisphereLight args={[0xffffff, 0xb9c6d2, 1.6]} />
      <directionalLight
        position={[-span, span * 2, span]}
        intensity={2.2}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-span}
        shadow-camera-right={span}
        shadow-camera-top={span}
        shadow-camera-bottom={-span}
      />
      <Model build={build} />
      <mesh rotation-x={-Math.PI / 2} receiveShadow>
        <circleGeometry args={[span * 1.4, 64]} />
        <shadowMaterial opacity={0.18} />
      </mesh>
      <OrbitControls target={target} makeDefault enableDamping maxPolarAngle={Math.PI * 0.55} />
    </Canvas>
  );
}
