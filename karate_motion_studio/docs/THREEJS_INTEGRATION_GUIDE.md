# Руководство по интеграции с Three.js и React (Bunkai Explorer)

В этом документе показано, как подключить экспортированные из **Karate Motion Studio** данные анимаций в существующий веб-проект **Bunkai Explorer** (React, `@react-three/fiber`, Three.js).

---

## Способ 1: Конвертация JSON в `THREE.AnimationClip` (Рекомендуемый)

Этот подход позволяет использовать стандартный `THREE.AnimationMixer`: кроссфейд между стойками, замедление (slow-motion), зацикливание и проигрывание нескольких техник подряд.

### Функция-конвертер `retargetedJsonToAnimationClip.js`

```javascript
import * as THREE from 'three';

/**
 * Преобразует bunkai-retargeted-animation JSON в THREE.AnimationClip
 * @param {Object} motionJson - Данные анимации из Karate Motion Studio
 * @param {String} rootBoneName - Имя корневой кости (по умолчанию "Hips")
 * @returns {THREE.AnimationClip}
 */
export function createAnimationClipFromRetargetedJson(motionJson, rootBoneName = 'Hips') {
  const { frames, fps, technique, duration } = motionJson;
  if (!frames || frames.length === 0) return null;

  const times = [];
  const rootPositions = [];
  const boneRotations = {};

  // Инициализация списков для всех костей
  for (const f of frames) {
    times.push(f.time);
    rootPositions.push(...f.rootPosition);

    for (const [boneName, quat] of Object.entries(f.bones)) {
      if (!boneRotations[boneName]) {
        boneRotations[boneName] = [];
      }
      boneRotations[boneName].push(...quat); // [x, y, z, w]
    }
  }

  const tracks = [];

  // 1. Трек перемещения корневой кости (Hips position)
  tracks.push(
    new THREE.VectorKeyframeTrack(
      `${rootBoneName}.position`,
      times,
      rootPositions
    )
  );

  // 2. Треки вращения каждой кости (Bone local quaternions)
  for (const [boneName, quats] of Object.entries(boneRotations)) {
    // Проверяем полноту трека (если кость была видна не на всех кадрах, дополняем)
    if (quats.length === times.length * 4) {
      tracks.push(
        new THREE.QuaternionKeyframeTrack(
          `${boneName}.quaternion`,
          times,
          quats
        )
      );
    }
  }

  return new THREE.AnimationClip(technique || 'karate_motion', duration, tracks);
}
```

---

## Способ 2: Компонент каратиста в React (`KaratekaAvatar.jsx`)

Использование с `@react-three/fiber` и `@react-three/drei`:

```jsx
import React, { useEffect, useRef, useState } from 'react';
import { useGLTF } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { createAnimationClipFromRetargetedJson } from './retargetedJsonToAnimationClip';

export function KaratekaAvatar({ 
  modelUrl = '/models/karateka_lowpoly.glb', 
  motionData, 
  isPlaying = true, 
  playbackRate = 1.0,
  onFinish 
}) {
  const group = useRef();
  const { scene } = useGLTF(modelUrl);
  const [mixer, setMixer] = useState(null);
  const actionRef = useRef(null);

  useEffect(() => {
    if (!scene || !motionData) return;

    // Создаем AnimationMixer для загруженного аватара
    const newMixer = new THREE.AnimationMixer(scene);
    const clip = createAnimationClipFromRetargetedJson(motionData);
    
    if (clip) {
      const action = newMixer.clipAction(clip);
      action.setLoop(THREE.LoopRepeat);
      action.clampWhenFinished = false;
      action.play();
      actionRef.current = action;
    }

    setMixer(newMixer);

    return () => {
      newMixer.stopAllAction();
    };
  }, [scene, motionData]);

  // Управление скоростью и воспроизведением
  useEffect(() => {
    if (actionRef.current) {
      actionRef.current.setEffectiveTimeScale(playbackRate);
      actionRef.current.paused = !isPlaying;
    }
  }, [isPlaying, playbackRate]);

  // Обновление кадров в анимационном цикле React Three Fiber
  useFrame((state, delta) => {
    if (mixer && isPlaying) {
      mixer.update(delta);
    }
  });

  return (
    <primitive 
      ref={group} 
      object={scene} 
      position={[0, 0, 0]} 
      rotation={[0, 0, 0]} 
    />
  );
}

useGLTF.preload('/models/karateka_lowpoly.glb');
```

---

## Способ 3: Интеграция в Bunkai Explorer Ката-плеер

Пример переключения техник и ката в интерфейсе:

```jsx
import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { KaratekaAvatar } from './KaratekaAvatar';

export function BunkaiViewer() {
  const [currentTechnique, setCurrentTechnique] = useState('gedan_barai');
  const [motionData, setMotionData] = useState(null);
  const [speed, setSpeed] = useState(1.0);
  const [isPlaying, setIsPlaying] = useState(true);

  // Загрузка экспортированного JSON из Karate Motion Studio
  useEffect(() => {
    fetch(`/motions/${currentTechnique}_anim.json`)
      .then((res) => res.json())
      .then((data) => setMotionData(data))
      .catch((err) => console.error("Ошибка загрузки анимации:", err));
  }, [currentTechnique]);

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Панель управления ката и техниками */}
      <div style={{ padding: '12px 20px', background: '#18191c', color: '#fff', display: 'flex', gap: '15px' }}>
        <button onClick={() => setCurrentTechnique('gedan_barai')}>Гедан-барай</button>
        <button onClick={() => setCurrentTechnique('oi_zuki')}>Ой-дзуки</button>
        <button onClick={() => setCurrentTechnique('zenkutsu_dachi')}>Дзенкуцу-дати</button>
        <button onClick={() => setIsPlaying(!isPlaying)}>{isPlaying ? 'Пауза' : 'Пуск'}</button>
        <label>
          Скорость:
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={0.25}>0.25x (Анализ)</option>
            <option value={0.5}>0.5x</option>
            <option value={1.0}>1.0x (Норма)</option>
          </select>
        </label>
      </div>

      {/* 3D Dojo Viewport */}
      <div style={{ flex: 1, background: '#121316' }}>
        <Canvas camera={{ position: [0, 1.4, 3.2], fov: 45 }}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[3, 5, 4]} intensity={1.2} castShadow />
          
          {/* Сетка татами */}
          <Grid 
            args={[10, 10]} 
            sectionColor="#404552" 
            cellColor="#26282e" 
            position={[0, 0, 0]} 
          />

          {/* 3D Аватар каратиста */}
          {motionData && (
            <KaratekaAvatar 
              motionData={motionData} 
              isPlaying={isPlaying} 
              playbackRate={speed} 
            />
          )}

          <OrbitControls target={[0, 0.9, 0]} makeDefault />
        </Canvas>
      </div>
    </div>
  );
}
```

---

## Способ 4: Прямое покадровое управление (Scrubber)

Если вам требуется синхронизировать видеозапись с 3D-моделью ползунком таймлайна (без AnimationMixer):

```javascript
export function applyFramePoseToArmature(armature, frameData) {
  if (!armature || !frameData) return;

  // 1. Позиция таза (Hips)
  const hips = armature.getObjectByName('Hips');
  if (hips && frameData.rootPosition) {
    hips.position.set(
      frameData.rootPosition[0],
      frameData.rootPosition[1],
      frameData.rootPosition[2]
    );
  }

  // 2. Повороты всех костей
  for (const [boneName, quat] of Object.entries(frameData.bones)) {
    const bone = armature.getObjectByName(boneName);
    if (bone) {
      bone.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
    }
  }
}
```
