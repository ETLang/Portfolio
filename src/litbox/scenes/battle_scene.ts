import { vec4 } from 'gl-matrix';
import { LitboxScene } from '../litbox_scene.ts';
import type { AnyLight, RaytracedObject, SceneObject, SceneSprite } from '../scene.ts';
import type { LitboxSceneRenderer } from '../../litbox_scene_renderer.ts';
import type { DenoiserTunables } from '../simulation.ts';

/**
 * See LitboxScene.getDenoiserTunables's doc comment. densityBlurFalloff explicitly pinned to the
 * default (not just inherited) - the UFOs/beams here read as solid-surfaced geometry, not a
 * cloudy/subsurface-scattering volume (contrast CornellSquareScene's much higher value), so the
 * default's fast falloff is the intended look, not a placeholder.
 */
const DENOISER_TUNABLES: Partial<DenoiserTunables> = {
    densityBlurFalloff: 0.001,
};

const SPAWN_INTERVAL_MIN_SECONDS = 0.8;
const SPAWN_INTERVAL_MAX_SECONDS = 1.2;
const CROSSING_TIME_MIN_SECONDS = 3;
const CROSSING_TIME_MAX_SECONDS = 6;
const MAX_ANGLE_VARIATION_DEGREES = 30;
const SPEED_MULTIPLIER = 0.8; // 20% slower than the crossing-time range alone would produce
// Fraction of simulation-area height, from each edge, UFOs avoid flying into - applied at both the
// bottom (avoids an awkward floor-hugging path) and the top (the denoiser's shadow there gets
// heavily exaggerated compared to elsewhere in the frame, so this also doubles as guaranteeing a
// UFO's trajectory never reaches the top edge).
const EDGE_MARGIN_FRACTION = 0.1;
const OFFSCREEN_MARGIN = 1.5; // world units beyond the simulation area's edge to spawn/despawn at
const WOBBLE_AMPLITUDE_MIN = 0.05;
const WOBBLE_AMPLITUDE_MAX = 0.15;
const WOBBLE_FREQUENCY_MIN_HZ = 0.6;
const WOBBLE_FREQUENCY_MAX_HZ = 1.4;
const FIRE_INTERVAL_MIN_SECONDS = 0.5;
const FIRE_INTERVAL_MAX_SECONDS = 1.25;
const FIRE_DURATION_MIN_SECONDS = 0.5;
const FIRE_DURATION_MAX_SECONDS = 1.5;
const LASER_FLICKER_FREQUENCY_HZ = 5;
// Each UFO schedules its own next-fire time independently, normalized so the combined rate stays
// population-invariant: interval scales up by the current UFO count (canceling the "more UFOs -> more
// simultaneous shots" effect that would otherwise come from simply reusing the single-shooter interval
// per-instance), then scaled back down by this multiplier so the combined rate ends up 30% higher than
// that normalized baseline.
const FIRE_FREQUENCY_MULTIPLIER = 1.3;

const HOVER_INTERVAL_MIN_SECONDS = 6;
const HOVER_INTERVAL_MAX_SECONDS = 8; // mean 7s - normalized by UFO count the same way FIRE_INTERVAL is, so the global rate stays ~once/7s regardless of population
const HOVER_DURATION_MIN_SECONDS = 2;
const HOVER_DURATION_MAX_SECONDS = 3;

// Fraction of laser hits that actually down the target - the rest land (beam fires, gimbal
// aims, visuals play) but the target flies on unaffected, so not every shot reads as a kill.
const UFO_SHOOTDOWN_CHANCE = 0.5;

const SHOCK_DURATION_SECONDS = 0.5;
const SHOCK_IMPULSE_DEGREES_MIN = 25;
const SHOCK_IMPULSE_DEGREES_MAX = 45;
const SHOCK_OSCILLATION_HZ = 3; // wobbles per second while correcting back toward level

const FALL_GRAVITY = 3; // world-units/second^2 downward acceleration once a hit UFO loses control
const FALL_SPIN_MIN_DEG_PER_SEC = 360;
const FALL_SPIN_MAX_DEG_PER_SEC = 720;

/** 6 high-saturation primary/secondary body colors a spawned UFO picks from. */
const BODY_COLORS: ReadonlyArray<{ r: number; g: number; b: number }> = [
    { r: 1, g: 0, b: 0 }, // red
    { r: 0, g: 1, b: 0 }, // green
    { r: 0, g: 0, b: 1 }, // blue
    { r: 1, g: 1, b: 0 }, // yellow
    { r: 0, g: 1, b: 1 }, // cyan
    { r: 1, g: 0, b: 1 }, // magenta
];

interface SimulationBounds {
    left: number;
    right: number;
    top: number;
    bottom: number;
}

/**
 * 'flying' - normal path-following (and possibly hovering/firing). 'shocked' - just hit, rotation
 * kicked and correcting, translation still following its normal path for SHOCK_DURATION_SECONDS.
 * 'falling' - lost control: gravity-driven fall + free spin, no more hovering/firing/targeting.
 */
type UfoState = 'flying' | 'shocked' | 'falling';

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
    /** This UFO's own next-fire time - independent per instance, see FIRE_FREQUENCY_MULTIPLIER. */
    nextFireTimeSeconds: number;

    hovering: boolean;
    /** This UFO's own next-hover-decision time - independent per instance, see HOVER_INTERVAL_MIN/MAX_SECONDS. */
    nextHoverTimeSeconds: number;
    hoverStartTimeSeconds: number;
    hoverDurationSeconds: number;
    /** Total seconds spent hovering so far (completed hovers only) - subtracted from elapsed spawn time to compute path position, so a hover freezes translation without perturbing the closed-form position formula. */
    pausedSecondsAccumulated: number;

    state: UfoState;
    shockStartTimeSeconds: number;
    /** Signed rotational impulse (degrees) applied at the moment of the hit; decays back toward 0 over SHOCK_DURATION_SECONDS. */
    shockImpulseDegrees: number;

    /** Position/velocity/rotation captured at the instant control was lost - the falling phase's own closed-form gravity/spin formulas are relative to these, not accumulated frame-to-frame. */
    fallStartTimeSeconds: number;
    fallStartX: number;
    fallStartY: number;
    fallStartVelocityY: number;
    fallStartRotationDegrees: number;
    fallSpinDegPerSec: number;
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

    public override getDenoiserTunables(): Partial<DenoiserTunables> {
        return DENOISER_TUNABLES;
    }

    public override onFrame(deltaTimeSeconds: number): void {
        this.sceneTimeSeconds += deltaTimeSeconds;

        const bounds = this.getSimulationBounds();
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
    }

    /**
     * The simulation area's world-space rectangle - UFOs spawn/despawn/avoid-the-floor relative
     * to this, not the camera's (possibly larger) full view, so anything the Crop overlay hides
     * outside the simulation area doesn't count as "still visible." Derived from the simulation
     * owner's current world transform (which maps its local [-0.5,0.5]^2 rect into world space -
     * see SimulationResources.getWorldTransform) rather than assumed, so it stays correct if that
     * transform ever changes.
     */
    private getSimulationBounds(): SimulationBounds | null {
        const simulationResources = this.renderer.getSimulationResources();
        if (!simulationResources.hasSimulation()) {
            return null;
        }
        const worldTransform = simulationResources.getWorldTransform();
        const topLeft = vec4.transformMat4(vec4.create(), vec4.fromValues(-0.5, 0.5, 0, 1), worldTransform);
        const bottomRight = vec4.transformMat4(vec4.create(), vec4.fromValues(0.5, -0.5, 0, 1), worldTransform);
        return { left: topLeft[0], right: bottomRight[0], top: topLeft[1], bottom: bottomRight[1] };
    }

    private spawnUfo(bounds: SimulationBounds): void {
        const fromLeft = Math.random() < 0.5;
        const crossingSeconds = randomRange(CROSSING_TIME_MIN_SECONDS, CROSSING_TIME_MAX_SECONDS);
        const areaWidth = bounds.right - bounds.left;
        const angleRadians = (randomRange(-MAX_ANGLE_VARIATION_DEGREES, MAX_ANGLE_VARIATION_DEGREES) * Math.PI) / 180;

        // horizontalSpeed alone crosses the simulation area in crossingSeconds; speed is scaled up
        // so its horizontal component still matches that, regardless of how steep angleRadians turned out.
        const horizontalSpeed = (areaWidth / crossingSeconds) * SPEED_MULTIPLIER;
        const speed = horizontalSpeed / Math.cos(angleRadians);
        const directionSign = fromLeft ? 1 : -1;
        const velocityX = directionSign * speed * Math.cos(angleRadians);
        const velocityY = speed * Math.sin(angleRadians);

        const floorY = bounds.bottom + EDGE_MARGIN_FRACTION * (bounds.top - bounds.bottom);
        const ceilingY = bounds.top - EDGE_MARGIN_FRACTION * (bounds.top - bounds.bottom);
        const spawnY = randomRange(floorY + 0.5, ceilingY - 0.5);
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

        // Includes the UFO being spawned right now - it schedules its own first fire/hover
        // relative to the population size it's joining.
        const population = this.ufos.length + 1;

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
            nextFireTimeSeconds: this.nextFireTimeSecondsFor(population),

            hovering: false,
            nextHoverTimeSeconds: this.nextHoverTimeSecondsFor(population),
            hoverStartTimeSeconds: 0,
            hoverDurationSeconds: 0,
            pausedSecondsAccumulated: 0,

            state: 'flying',
            shockStartTimeSeconds: 0,
            shockImpulseDegrees: 0,

            fallStartTimeSeconds: 0,
            fallStartX: 0,
            fallStartY: 0,
            fallStartVelocityY: 0,
            fallStartRotationDegrees: 0,
            fallSpinDegPerSec: 0,
        });
    }

    /** Per-UFO next-fire time, normalized so the combined firing rate stays population-invariant (times FIRE_FREQUENCY_MULTIPLIER) - see FIRE_FREQUENCY_MULTIPLIER's doc comment. */
    private nextFireTimeSecondsFor(population: number): number {
        return (
            this.sceneTimeSeconds +
            (randomRange(FIRE_INTERVAL_MIN_SECONDS, FIRE_INTERVAL_MAX_SECONDS) * population) / FIRE_FREQUENCY_MULTIPLIER
        );
    }

    /** Per-UFO next-hover-decision time, normalized so the combined hover rate stays ~once/HOVER_INTERVAL regardless of population. */
    private nextHoverTimeSecondsFor(population: number): number {
        return this.sceneTimeSeconds + randomRange(HOVER_INTERVAL_MIN_SECONDS, HOVER_INTERVAL_MAX_SECONDS) * population;
    }

    private updateUfo(ufo: UfoInstance, nowSeconds: number, bounds: SimulationBounds): void {
        if (ufo.state === 'falling') {
            this.updateFalling(ufo, nowSeconds);
            return;
        }

        if (ufo.state === 'flying') {
            this.updateHover(ufo, nowSeconds);
        }
        this.updateFlightPosition(ufo, nowSeconds, bounds);

        if (ufo.state === 'shocked') {
            const shockElapsed = nowSeconds - ufo.shockStartTimeSeconds;
            if (shockElapsed >= SHOCK_DURATION_SECONDS) {
                this.startFalling(ufo, nowSeconds);
                return;
            }
            ufo.root.rotation = this.computeShockRotationDegrees(ufo, shockElapsed);
            this.markTransformDirty(ufo.root);
            return;
        }

        this.updateFiringState(ufo, nowSeconds);
    }

    /**
     * Drives ufo.root's position for the 'flying' and 'shocked' states (rotation is untouched
     * here - 'shocked' overlays its own rotation on top, in updateUfo). A hovering UFO's
     * translation is frozen by subtracting accumulated hover time from elapsed before it ever
     * reaches the position formulas below, rather than branching the formulas themselves - so a
     * hover simply "pauses the clock" that spawnX/spawnY/velocity are measured against. Wobble
     * keeps running on true elapsed time regardless, so a hovering UFO still looks alive.
     */
    private updateFlightPosition(ufo: UfoInstance, nowSeconds: number, bounds: SimulationBounds): void {
        const floorY = bounds.bottom + EDGE_MARGIN_FRACTION * (bounds.top - bounds.bottom);
        const ceilingY = bounds.top - EDGE_MARGIN_FRACTION * (bounds.top - bounds.bottom);
        const trueElapsed = nowSeconds - ufo.spawnTimeSeconds;
        const pausedSoFar = ufo.pausedSecondsAccumulated + (ufo.hovering ? nowSeconds - ufo.hoverStartTimeSeconds : 0);
        const pathElapsed = trueElapsed - pausedSoFar;

        ufo.root.position.x = ufo.spawnX + ufo.velocityX * pathElapsed;
        // Once velocityY carries the unclamped term past floorY/ceilingY, the max()/min() latches
        // there for every later elapsed too (the unclamped term only keeps moving further past it) -
        // same "hits the edge and stays" behavior as an old per-frame clamp, but derived directly
        // from elapsed time instead of depending on last frame's already-clamped baseY.
        const baseY = Math.min(Math.max(ufo.spawnY + ufo.velocityY * pathElapsed, floorY), ceilingY);
        const wobblePhase = ufo.wobblePhase0 + ufo.wobbleFrequencyHz * 2 * Math.PI * trueElapsed;
        ufo.root.position.y = baseY + ufo.wobbleAmplitude * Math.sin(wobblePhase);
        this.markTransformDirty(ufo.root);
    }

    /** Starts/ends a hover independently per UFO - see HOVER_INTERVAL_MIN/MAX_SECONDS. Only called while ufo.state === 'flying'. */
    private updateHover(ufo: UfoInstance, nowSeconds: number): void {
        if (ufo.hovering) {
            if (nowSeconds - ufo.hoverStartTimeSeconds >= ufo.hoverDurationSeconds) {
                ufo.pausedSecondsAccumulated += ufo.hoverDurationSeconds;
                ufo.hovering = false;
                ufo.nextHoverTimeSeconds = this.nextHoverTimeSecondsFor(this.ufos.length);
            }
            return;
        }
        if (nowSeconds >= ufo.nextHoverTimeSeconds) {
            ufo.hovering = true;
            ufo.hoverStartTimeSeconds = nowSeconds;
            ufo.hoverDurationSeconds = randomRange(HOVER_DURATION_MIN_SECONDS, HOVER_DURATION_MAX_SECONDS);
        }
    }

    /** Tracks an in-progress burst (aiming/flicker/expiry), or starts a new one independently for this UFO - see FIRE_FREQUENCY_MULTIPLIER. Only called while ufo.state === 'flying'. */
    private updateFiringState(ufo: UfoInstance, nowSeconds: number): void {
        if (ufo.firing) {
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
            return;
        }

        if (nowSeconds < ufo.nextFireTimeSeconds) {
            return;
        }
        ufo.nextFireTimeSeconds = this.nextFireTimeSecondsFor(this.ufos.length);
        this.tryStartFiring(ufo, nowSeconds);
    }

    /** Picks a target (any other still-'flying' UFO) and starts a burst from `shooter`, immediately landing a hit on the target. No-op if no eligible target exists (e.g. fewer than 2 UFOs alive). */
    private tryStartFiring(shooter: UfoInstance, nowSeconds: number): void {
        const targets = this.ufos.filter(u => u !== shooter && u.state === 'flying');
        if (targets.length === 0) {
            return;
        }
        const target = targets[Math.floor(Math.random() * targets.length)];

        shooter.firingTarget = target;
        this.aimGimbalAt(shooter, target);
        shooter.laser.active = true;
        this.markTransformDirty(shooter.laser);
        shooter.firing = true;
        shooter.fireStartTimeSeconds = nowSeconds;
        shooter.fireDurationSeconds = randomRange(FIRE_DURATION_MIN_SECONDS, FIRE_DURATION_MAX_SECONDS);

        if (Math.random() < UFO_SHOOTDOWN_CHANCE) {
            this.applyHit(target, nowSeconds);
        }
    }

    /** The beam is instant (light-speed, and the gimbal is already aimed exactly at the target), so the hit lands the same frame the burst starts - kicks the target into 'shocked'. */
    private applyHit(target: UfoInstance, nowSeconds: number): void {
        if (target.hovering) {
            target.pausedSecondsAccumulated += nowSeconds - target.hoverStartTimeSeconds;
            target.hovering = false;
        }
        if (target.firing) {
            target.firing = false;
            target.firingTarget = null;
            target.laser.active = false;
            this.markTransformDirty(target.laser);
        }
        target.state = 'shocked';
        target.shockStartTimeSeconds = nowSeconds;
        target.shockImpulseDegrees = (Math.random() < 0.5 ? -1 : 1) * randomRange(SHOCK_IMPULSE_DEGREES_MIN, SHOCK_IMPULSE_DEGREES_MAX);
    }

    /** A damped oscillation from shockImpulseDegrees back toward 0 - the "rock from the hit, then correct" motion - reaching a small residual by SHOCK_DURATION_SECONDS, at which point startFalling takes over. */
    private computeShockRotationDegrees(ufo: UfoInstance, shockElapsed: number): number {
        const decay = Math.exp(-shockElapsed / (SHOCK_DURATION_SECONDS / 3));
        return ufo.shockImpulseDegrees * decay * Math.cos(2 * Math.PI * SHOCK_OSCILLATION_HZ * shockElapsed);
    }

    /** Transitions a UFO to 'falling': captures its current position/velocity/rotation as the origin for the closed-form gravity-fall + free-spin formulas in updateFalling. */
    private startFalling(ufo: UfoInstance, nowSeconds: number): void {
        ufo.state = 'falling';
        ufo.fallStartTimeSeconds = nowSeconds;
        ufo.fallStartX = ufo.root.position.x;
        ufo.fallStartY = ufo.root.position.y;
        ufo.fallStartVelocityY = ufo.velocityY;
        ufo.fallStartRotationDegrees = ufo.root.rotation;
        ufo.fallSpinDegPerSec = (Math.random() < 0.5 ? -1 : 1) * randomRange(FALL_SPIN_MIN_DEG_PER_SEC, FALL_SPIN_MAX_DEG_PER_SEC);
    }

    /** Gravity-accelerated fall + constant free spin, out of control - no floor clamp, no hovering, no firing. despawnExited removes it once it falls past the bottom edge. */
    private updateFalling(ufo: UfoInstance, nowSeconds: number): void {
        const fallElapsed = nowSeconds - ufo.fallStartTimeSeconds;
        ufo.root.position.x = ufo.fallStartX + ufo.velocityX * fallElapsed;
        ufo.root.position.y = ufo.fallStartY + ufo.fallStartVelocityY * fallElapsed - 0.5 * FALL_GRAVITY * fallElapsed * fallElapsed;
        ufo.root.rotation = ufo.fallStartRotationDegrees + ufo.fallSpinDegPerSec * fallElapsed;
        this.markTransformDirty(ufo.root);
    }

    private despawnExited(bounds: SimulationBounds): void {
        const survivors: UfoInstance[] = [];
        for (const ufo of this.ufos) {
            const exitedRight = ufo.velocityX > 0 && ufo.root.position.x > bounds.right + OFFSCREEN_MARGIN;
            const exitedLeft = ufo.velocityX < 0 && ufo.root.position.x < bounds.left - OFFSCREEN_MARGIN;
            const exitedBottom = ufo.state === 'falling' && ufo.root.position.y < bounds.bottom - OFFSCREEN_MARGIN;
            if (exitedRight || exitedLeft || exitedBottom) {
                this.destroyObject(ufo.root);
            } else {
                survivors.push(ufo);
            }
        }
        this.ufos = survivors;
    }

    /**
     * Rotates `shooter.gimbal` so the laser beam points at `target`'s current position - called
     * once when a burst starts and again every frame for the rest of the burst, so the beam tracks
     * a moving target instead of freezing on its position when the burst began.
     */
    private aimGimbalAt(shooter: UfoInstance, target: UfoInstance): void {
        // Gimbal world position - a plain scaled offset, since the shooter is always still
        // 'flying' here (root.rotation === 0) even though a 'shocked'/'falling' *target*'s root
        // may be rotated - aimGimbalAt only ever reads target.root.position, never its rotation.
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
