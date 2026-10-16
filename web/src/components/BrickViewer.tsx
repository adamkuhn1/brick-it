import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { buildBrickScene, stepOrder } from "../three/bricks";
import type { Build } from "../types";
import StepScrubber from "./StepScrubber";

function Model({ build, visible }: { build: Build; visible: number }) {
  const material = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.32, metalness: 0 }), []);
  const scene = useMemo(() => {
    const s = buildBrickScene(build, stepOrder(build), material);
    s.group.traverse((o) => {
      o.castShadow = true;
      o.receiveShadow = true;
    });
    return s;
  }, [build, material]);
  useEffect(() => () => scene.dispose(), [scene]);
  useEffect(() => {
    scene.setVisible(visible);
  }, [scene, visible]);
  return <primitive object={scene.group} />;
}

export default function BrickViewer({ build }: { build: Build }) {
  const span = Math.max(build.dims.x, build.dims.y, build.dims.z * 1.2);
  const target: [number, number, number] = [0, (build.dims.z * 1.2) / 2, 0];
  // Bricks placed; a new build (e.g. after a size change) starts on the finished model.
  const [shown, setShown] = useState({ build, k: build.bricks.length });
  const k = shown.build === build ? shown.k : build.bricks.length;
  return (
    <>
      <Canvas
        className="viewer-canvas"
        style={{ height: "auto" }}
        shadows
        dpr={[1, 2]}
        camera={{ position: [-span * 1.5, span * 1.35, span * 1.9], fov: 35, near: 0.1, far: span * 20 }}
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
        <Model build={build} visible={k} />
        <mesh rotation-x={-Math.PI / 2} receiveShadow>
          <circleGeometry args={[span * 1.4, 64]} />
          <shadowMaterial opacity={0.18} />
        </mesh>
        <OrbitControls target={target} makeDefault enableDamping maxPolarAngle={Math.PI * 0.55} />
      </Canvas>
      <StepScrubber build={build} value={k} onChange={(n) => setShown({ build, k: n })} />
    </>
  );
}
