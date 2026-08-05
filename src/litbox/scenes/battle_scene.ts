import { LitboxScene } from '../litbox_scene.ts';
import type { AnyLight, RaytracedObject, SceneObject, SceneSprite } from '../scene.ts';
import type { LitboxSceneRenderer } from '../../litbox_scene_renderer.ts';

const SPAWN_INTERVAL_MIN_SECONDS = 0.8;
const SPAWN_INTERVAL_MAX_SECONDS = 1.2;
const CROSSING_TIME_MIN_SECONDS = 3;
const CROSSING_TIME_MAX_SECONDS = 6;
const MAX_ANGLE_VARIATION_DEGREES = 30;
const SPEED_MULTIPLIER = 0.8; // 20% slower than the crossing-time range alone would produce
const FLOOR_MARGIN_FRACTION = 0.1; // fraction of screen height, from the bottom, UFOs avoid flying into
const OFFSCREEN_MARGIN = 1.5; // world units beyond the screen edge to spawn/despawn at
const WOBBLE_AMPLITUDE_MIN = 0.05;
const WOBBLE_AMPLITUDE_MAX = 0.15;
const WOBBLE_FREQUENCY_MIN_HZ = 0.6;
const WOBBLE_FREQUENCY_MAX_HZ = 1.4;
const FIRE_INTERVAL_MIN_SECONDS = 1;
const FIRE_INTERVAL_MAX_SECONDS = 2;
const FIRE_DURATION_MIN_SECONDS = 0.5;
const FIRE_DURATION_MAX_SECONDS = 1.5;
const LASER_FLICKER_FREQUENCY_HZ = 5;

/** 6 high-saturation primary/secondary body colors a spawned UFO picks from. */
const BODY_COLORS: ReadonlyArray<{ r: number; g: number; b: number }> = [
    { r: 1, g: 0, b: 0 }, // red
    { r: 0, g: 1, b: 0 }, // green
    { r: 0, g: 0, b: 1 }, // blue
    { r: 1, g: 1, b: 0 }, // yellow
    { r: 0, g: 1, b: 1 }, // cyan
    { r: 1, g: 0, b: 1 }, // magenta
];

interface ScreenBounds {
    left: number;
    right: number;
    top: number;
    bottom: number;
}

interface UfoInstance {
    root: SceneObject;
    gimbal: SceneObject;
    laser: SceneObject;
    laserLight: AnyLight;
    bodySprite: SceneSprite;
    bodyRaytraced: RaytracedObject;
    velocityX: number;
    velocityY: number;
    /** Position/phase at spawn - every per-frame value is derived from these plus elapsed scene time, never accumulated frame-to-frame. */
    spawnTimeSeconds: number;
    spawnX: number;
    /** Pre-wobble Y position at spawn - the floor-avoidance clamp applies to this, never to the wobbled render position. */
    spawnY: number;
    wobbleAmplitude: number;
    wobbleFrequencyHz: number;
    wobblePhase0: number;
    firing: boolean;
    /** The UFO this one is currently shooting at - re-aimed at every frame while firing so the beam tracks its movement. Null while not firing. */
    firingTarget: UfoInstance | null;
    fireStartTimeSeconds: number;
    fireDurationSeconds: number;
}

function randomRange(min: number, max: number): number {
    return min + Math.random() * (max - min);
}

/** Scene-specific animation/interaction logic for battle.json. */
export class BattleScene extends LitboxScene {
    public static readonly jsonPath = 'scenes/battle.json';

    private renderer!: LitboxSceneRenderer;
    private ufoTemplate!: SceneObject;
    private baseLaserIntensity = 0;

    private ufos: UfoInstance[] = [];
    /** Elapsed wall-clock seconds since the scene loaded - the single "current time" every animation below is computed relative to, rather than each animation accumulating its own independent per-frame state. */
    private sceneTimeSeconds = 0;
    private nextSpawnTimeSeconds = randomRange(SPAWN_INTERVAL_MIN_SECONDS, SPAWN_INTERVAL_MAX_SECONDS);
    private nextFireTimeSeconds = randomRange(FIRE_INTERVAL_MIN_SECONDS, FIRE_INTERVAL_MAX_SECONDS);

    public override onLoad(renderer: LitboxSceneRenderer): void {
        this.setActiveCamera('Main Camera');
        this.renderer = renderer;

        // The UFO object in the scene JSON is the template every spawned instance is cloned
        // from - deactivate it so the template itself is never rendered.
        this.ufoTemplate = this.getObject('UFO');
        this.ufoTemplate.active = false;

        const templateLaser = this.resolveRelativePath(this.ufoTemplate, 'Laser Gimbal/Laser');
        this.baseLaserIntensity = this.getLight(templateLaser).intensity;
    }

    public override onFrame(deltaTimeSeconds: number): void {
        this.sceneTimeSeconds += deltaTimeSeconds;

        const bounds = this.getScreenBounds();
        if (!bounds) {
            return;
        }

        if (this.sceneTimeSeconds >= this.nextSpawnTimeSeconds) {
            this.nextSpawnTimeSeconds =
                this.sceneTimeSeconds + randomRange(SPAWN_INTERVAL_MIN_SECONDS, SPAWN_INTERVAL_MAX_SECONDS);
            this.spawnUfo(bounds);
        }

        for (const ufo of this.ufos) {
            this.updateUfo(ufo, this.sceneTimeSeconds, bounds);
        }
        this.despawnExited(bounds);

        if (this.sceneTimeSeconds >= this.nextFireTimeSeconds) {
            this.nextFireTimeSeconds =
                this.sceneTimeSeconds + randomRange(FIRE_INTERVAL_MIN_SECONDS, FIRE_INTERVAL_MAX_SECONDS);
            this.tryStartRandomFire();
        }
    }

    /** The active camera's visible world-space rectangle, derived from the canvas size rather than assumed, so it stays correct across window resizes. */
    private getScreenBounds(): ScreenBounds | null {
        const canvas = this.renderer.getCanvas();
        const topLeft = this.renderer.screenToWorld(0, 0);
        const bottomRight = this.renderer.screenToWorld(canvas.width, canvas.height);
        if (!topLeft || !bottomRight) {
            return null;
        }
        return { left: topLeft.x, right: bottomRight.x, top: topLeft.y, bottom: bottomRight.y };
    }

    private spawnUfo(bounds: ScreenBounds): void {
        const fromLeft = Math.random() < 0.5;
        const crossingSeconds = randomRange(CROSSING_TIME_MIN_SECONDS, CROSSING_TIME_MAX_SECONDS);
        const screenWidth = bounds.right - bounds.left;
        const angleRadians = (randomRange(-MAX_ANGLE_VARIATION_DEGREES, MAX_ANGLE_VARIATION_DEGREES) * Math.PI) / 180;

        // horizontalSpeed alone crosses the screen in crossingSeconds; speed is scaled up so its
        // horizontal component still matches that, regardless of how steep angleRadians turned out.
        const horizontalSpeed = (screenWidth / crossingSeconds) * SPEED_MULTIPLIER;
        const speed = horizontalSpeed / Math.cos(angleRadians);
        const directionSign = fromLeft ? 1 : -1;
        const velocityX = directionSign * speed * Math.cos(angleRadians);
        const velocityY = speed * Math.sin(angleRadians);

        const floorY = bounds.bottom + FLOOR_MARGIN_FRACTION * (bounds.top - bounds.bottom);
        const spawnY = randomRange(floorY + 0.5, bounds.top - 0.5);
        const spawnX = fromLeft ? bounds.left - OFFSCREEN_MARGIN : bounds.right + OFFSCREEN_MARGIN;

        const root = this.cloneObject(this.ufoTemplate, {
            position: { x: spawnX, y: spawnY },
            active: true,
        });

        const gimbal = this.resolveRelativePath(root, 'Laser Gimbal');
        const laser = this.resolveRelativePath(gimbal, 'Laser');
        laser.active = false;

        const bodySprite = this.getSprite(this.resolveRelativePath(root, 'Bottom/Sprite'));
        const bodyRaytraced = this.getRaytraced(this.resolveRelativePath(root, 'Bottom/Traced'));
        const color = BODY_COLORS[Math.floor(Math.random() * BODY_COLORS.length)];
        bodySprite.colorMod = { ...bodySprite.colorMod, r: color.r, g: color.g, b: color.b };
        bodyRaytraced.albedo = { ...bodyRaytraced.albedo, r: color.r, g: color.g, b: color.b };
        this.markSpriteDirty(bodySprite);
        this.markRayTracedDirty(bodyRaytraced);

        const laserLight = this.getLight(laser);

        this.ufos.push({
            root,
            gimbal,
            laser,
            laserLight,
            bodySprite,
            bodyRaytraced,
            velocityX,
            velocityY,
            spawnTimeSeconds: this.sceneTimeSeconds,
            spawnX,
            spawnY,
            wobbleAmplitude: randomRange(WOBBLE_AMPLITUDE_MIN, WOBBLE_AMPLITUDE_MAX),
            wobbleFrequencyHz: randomRange(WOBBLE_FREQUENCY_MIN_HZ, WOBBLE_FREQUENCY_MAX_HZ),
            wobblePhase0: Math.random() * Math.PI * 2,
            firing: false,
            firingTarget: null,
            fireStartTimeSeconds: 0,
            fireDurationSeconds: 0,
        });
    }

    private updateUfo(ufo: UfoInstance, nowSeconds: number, bounds: ScreenBounds): void {
        const floorY = bounds.bottom + FLOOR_MARGIN_FRACTION * (bounds.top - bounds.bottom);
        const elapsed = nowSeconds - ufo.spawnTimeSeconds;

        ufo.root.position.x = ufo.spawnX + ufo.velocityX * elapsed;
        // Once velocityY < 0 carries baseY below floorY, this max() latches at floorY for every
        // later elapsed too (the unclamped term only keeps decreasing) - same "hits the floor and
        // stays" behavior as the old per-frame clamp, but derived directly from elapsed time instead
        // of depending on last frame's already-clamped baseY.
        const baseY = Math.max(ufo.spawnY + ufo.velocityY * elapsed, floorY);
        const wobblePhase = ufo.wobblePhase0 + ufo.wobbleFrequencyHz * 2 * Math.PI * elapsed;
        ufo.root.position.y = baseY + ufo.wobbleAmplitude * Math.sin(wobblePhase);
        this.markTransformDirty(ufo.root);

        if (!ufo.firing) {
            return;
        }
        const fireElapsed = nowSeconds - ufo.fireStartTimeSeconds;
        if (fireElapsed >= ufo.fireDurationSeconds) {
            ufo.firing = false;
            ufo.firingTarget = null;
            ufo.laser.active = false;
            this.markTransformDirty(ufo.laser);
            return;
        }
        if (ufo.firingTarget) {
            this.aimGimbalAt(ufo, ufo.firingTarget);
        }
        // Oscillates between baseLaserIntensity/2 (low) and baseLaserIntensity (high, the template's own value).
        const oscillation = 0.5 + 0.5 * Math.sin(2 * Math.PI * LASER_FLICKER_FREQUENCY_HZ * fireElapsed);
        ufo.laserLight.intensity = (this.baseLaserIntensity / 2) * (1 + oscillation);
        this.markLightDirty(ufo.laserLight);
    }

    private despawnExited(bounds: ScreenBounds): void {
        const survivors: UfoInstance[] = [];
        for (const ufo of this.ufos) {
            const exitedRight = ufo.velocityX > 0 && ufo.root.position.x > bounds.right + OFFSCREEN_MARGIN;
            const exitedLeft = ufo.velocityX < 0 && ufo.root.position.x < bounds.left - OFFSCREEN_MARGIN;
            if (exitedRight || exitedLeft) {
                this.destroyObject(ufo.root);
            } else {
                survivors.push(ufo);
            }
        }
        this.ufos = survivors;
    }

    private tryStartRandomFire(): void {
        if (this.ufos.length < 2) {
            return;
        }
        const eligibleShooters = this.ufos.filter(u => !u.firing);
        if (eligibleShooters.length === 0) {
            return;
        }
        const shooter = eligibleShooters[Math.floor(Math.random() * eligibleShooters.length)];
        const targets = this.ufos.filter(u => u !== shooter);
        const target = targets[Math.floor(Math.random() * targets.length)];

        shooter.firingTarget = target;
        this.aimGimbalAt(shooter, target);

        shooter.laser.active = true;
        this.markTransformDirty(shooter.laser);
        shooter.firing = true;
        shooter.fireStartTimeSeconds = this.sceneTimeSeconds;
        shooter.fireDurationSeconds = randomRange(FIRE_DURATION_MIN_SECONDS, FIRE_DURATION_MAX_SECONDS);
    }

    /**
     * Rotates `shooter.gimbal` so the laser beam points at `target`'s current position - called
     * once when a burst starts and again every frame for the rest of the burst, so the beam tracks
     * a moving target instead of freezing on its position when the burst began.
     */
    private aimGimbalAt(shooter: UfoInstance, target: UfoInstance): void {
        // Gimbal world position (UFO itself never rotates, so this is a plain scaled offset).
        const gimbalWorldX = shooter.root.position.x + shooter.gimbal.position.x * shooter.root.scale.x;
        const gimbalWorldY = shooter.root.position.y + shooter.gimbal.position.y * shooter.root.scale.y;
        const angleToTargetDegrees =
            (Math.atan2(target.root.position.y - gimbalWorldY, target.root.position.x - gimbalWorldX) * 180) / Math.PI;
        // The Laser's own rotation (fixed, relative to the gimbal) is the template's baked beam
        // offset - the gimbal's rotation is the only aim adjustment we make. forward_monte_carlo.wgsl's
        // LIGHT_KIND_LASER emitter fires along the Laser object's local -Y axis (not +X - its scale,
        // 0.05 wide by 0.142 tall, is a tall thin beam shape), which points at world angle
        // (totalRotation + 90deg); solving totalRotation = gimbal.rotation + laser.rotation for the
        // beam to point at the target adds that same 90deg here.
        shooter.gimbal.rotation = angleToTargetDegrees + 90 - shooter.laser.rotation;
        this.markTransformDirty(shooter.gimbal);
    }
}
