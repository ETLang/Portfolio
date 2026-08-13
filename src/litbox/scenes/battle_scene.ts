import { mat4, vec4 } from 'gl-matrix';
import { LitboxScene, type BoundingBox } from '../litbox_scene.ts';
import type { AnyLight, Color, RaytracedObject, SceneObject, SceneSprite, Vector2 } from '../scene.ts';
import type { LitboxSceneRenderer } from '../../litbox_scene_renderer.ts';
import type { DenoiserTunables } from '../simulation.ts';
import { testSegmentAgainstShape, type PrimitiveShape } from '../collision.ts';

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

// Fraction of the simulation area's height (0 = bottom edge, 1 = top edge) a Cloud_N is allowed to
// re-spawn at once it drifts out of view - kept off both the very top and bottom of the sky.
const CLOUD_HEIGHT_FRACTION_MIN = 0.4;
const CLOUD_HEIGHT_FRACTION_MAX = 0.9;
// Cloud drift speed is a function of CLOUD_HEIGHT_FRACTION - a cheap parallax cue (higher/further
// clouds move slower) without an actual depth axis. World-units/second.
const CLOUD_SPEED_AT_LOW_ALTITUDE = 0.35; // at CLOUD_HEIGHT_FRACTION_MIN - closest, fastest
const CLOUD_SPEED_AT_HIGH_ALTITUDE = 0.12; // at CLOUD_HEIGHT_FRACTION_MAX - furthest, slowest
// Haze_L_N/Haze_R_N drift at a constant pace (world-units/second) - Haze_L toward -X, Haze_R toward +X.
const HAZE_SPEED = 0.2;

const TURRET_FIRE_RATE_HZ = 15; // 50% more frequent than the original 10
// How long, once firing starts/stops, the spread ramp takes to reach its max/settle back to 0 -
// see BattleScene.spreadFactorNow's doc comment for why this is a snapshot-plus-elapsed-time
// formula rather than an accumulated per-frame delta.
const TURRET_SPREAD_RAMP_UP_SECONDS = 0.5;
const TURRET_SPREAD_RAMP_DOWN_SECONDS = 0.5;
const TURRET_SPREAD_MAX_DEGREES = 6; // jitter applied to a fired round's angle, scaled by the current spread factor (0-1)
const TRACER_SPEED = 6.5; // world-units/second
const TRACER_SPEED_JITTER_FRACTION = 0.05; // small, constant (non-ramping) per-shot speed variance
const TRACER_GRAVITY = 2; // world-units/second^2
const TRACER_STRETCH_PER_SPEED = 0.7; // additional root.scale.y per world-unit/second of instantaneous speed
// A tracer despawns once it's burned this long, regardless of whether it's still in flight -
// see updateTracerBrightness's envelope, which fades linearly to 0 over exactly this span.
const TRACER_LIFETIME_SECONDS = 3;
// Flicker amplitude (fraction of brightness) ramps from 0 at spawn to this maximum as the tracer
// nears the end of its life - a dying flare's burn gets less steady, not more, as it runs low on
// fuel. Two incommensurate sine terms (frequency/phase fixed once at spawn - see fireTracer) stand
// in for irregular flicker, matching this file's "derive from elapsed time" convention rather than
// true per-frame randomness.
const TRACER_FLICKER_MAX_AMPLITUDE = 0.8;
const TRACER_FLICKER_FREQUENCY_MIN_HZ = 8;
const TRACER_FLICKER_FREQUENCY_MAX_HZ = 20;
// Fraction of tracer hits that actually down the target UFO - a hit always consumes the bullet,
// but (mirroring UFO_SHOOTDOWN_CHANCE) not every impact reads as a kill, since UFOs have no HP pool.
const TRACER_HIT_DOWN_CHANCE = 0.5;

// Searchlight_N lazily sweeps around its own authored resting angle (captured as scanCenterDegrees
// at discovery) and only ever notices a UFO that actually crosses through its beam - the narrow
// cone dead ahead of wherever the beam currently, physically points (SEARCHLIGHT_DETECTION_HALF_ANGLE_DEGREES),
// not any 'flying' UFO scene-wide regardless of where the beam happens to be aimed. Each such
// crossing (the UFO entering that cone, tracked per-instance via ufosInBeam so it's a one-shot
// event on entry, not re-rolled every frame it lingers) is an independent coin flip
// (SEARCHLIGHT_ACQUIRE_CHANCE) for whether the light bothers to lock onto it. Once locked, it
// tracks that UFO for as long as it stays 'flying' - only a destroyed/exited target releases the
// lock, back to lazy scanning (and a cleared ufosInBeam, so re-acquiring needs a fresh crossing,
// not credit for whatever already happened to be sitting in the cone at release time). Whatever the
// current desired aim is (scanning or locked), the beam only ever turns toward it at a bounded rate
// (see SEARCHLIGHT_MAX_TURN_RATE_DEGREES_PER_SECOND and updateSearchlights' use of stepAngleTowards)
// rather than snapping straight to it - a real searchlight is physically swung by a motor, so its
// rotation can never simply teleport ("pop") to a new angle, whether that's from scanning to a
// freshly acquired lock, from one target to a farther-away replacement, or across the +-180 degree
// wraparound seam.
const SEARCHLIGHT_SCAN_AMPLITUDE_MIN_DEGREES = 15;
const SEARCHLIGHT_SCAN_AMPLITUDE_MAX_DEGREES = 30;
const SEARCHLIGHT_SCAN_FREQUENCY_MIN_HZ = 0.05;
const SEARCHLIGHT_SCAN_FREQUENCY_MAX_HZ = 0.12;
// Half-width of the beam's detection cone - a UFO more than this many degrees off the beam's
// current aim isn't "in front of it" yet, regardless of how interesting a target it'd make.
const SEARCHLIGHT_DETECTION_HALF_ANGLE_DEGREES = 6;
// Chance a crossing UFO actually gets noticed and locked onto, rolled once per crossing (see
// ufosInBeam above) rather than continuously - a miss just means this particular light didn't
// happen to catch that particular pass, not that it stops looking.
const SEARCHLIGHT_ACQUIRE_CHANCE = 0.25;
// Still faster than the scan sweep's own max angular speed (amplitude * 2*pi*frequency tops out
// around 23deg/s at this range's upper bounds), so a lock-on still reads as turning toward its
// target rather than lagging behind it, but slow enough to read as a heavy, motor-driven swing
// rather than a snap - a UFO can still outrun a locked beam's turn rate before it reaches the
// UFO's current bearing, letting a fast enough target visibly pull away instead of being pinned.
const SEARCHLIGHT_MAX_TURN_RATE_DEGREES_PER_SECOND = 40;

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
 * A Cloud_N object drifting left across the sky. `localBounds` is computed once (LitboxScene.
 * computeLocalBounds never changes for a subtree whose local transforms are never touched) and
 * reused for both the "has it exited" test and placing a respawned cloud just off-screen.
 * `velocityX` is always <= 0, re-derived from a freshly-randomized height every respawn - see
 * cloudSpeedForHeightFraction.
 *
 * `worldX`/`worldY` (not `root.position`) are the authoritative position state: Cloud_N is
 * parented under a rig object that itself carries a nonzero world offset, so `root.position` (its
 * *local*, parent-relative position) isn't directly comparable to `bounds`, which
 * getSimulationBounds() computes in true world space. worldX/worldY are updated directly and
 * only ever pushed into `root.position` via LitboxScene.setWorldPosition - see its doc comment.
 */
interface CloudInstance {
    root: SceneObject;
    localBounds: BoundingBox;
    worldX: number;
    worldY: number;
    velocityX: number;
}

/**
 * A Haze_L_N/Haze_R_N band drifting at a constant pace - `velocityX`'s sign (fixed at
 * construction: negative for Haze_L, positive for Haze_R) is what the update loop below branches
 * on, so both groups share one code path. worldX/worldY: see CloudInstance's doc comment.
 */
interface HazeInstance {
    root: SceneObject;
    localBounds: BoundingBox;
    worldX: number;
    worldY: number;
    velocityX: number;
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
    /** The "Bottom/Traced" object itself (not just its RaytracedObject data) - needed by tracer collision to compute this UFO's current world transform. */
    bodyObject: SceneObject;
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

/**
 * A single fired round - position is a closed-form function of elapsed time from spawn (gravity
 * only touches the Y term, so velocityYAtSpawn - TRACER_GRAVITY * t is still closed-form, no
 * accumulation needed), matching every other per-frame-derived state in this file.
 */
interface TracerInstance {
    root: SceneObject;
    sprite: SceneSprite;
    light: AnyLight;
    /** Emissive color/light intensity as originally authored on this clone - brightness scales these down, never overwrites them, so repeated frames don't compound. */
    baseEmissive: Color;
    baseLightIntensity: number;
    spawnTimeSeconds: number;
    spawnX: number;
    spawnY: number;
    velocityX: number;
    velocityYAtSpawn: number;
    /** True once this tracer has been observed inside the simulation bounds at least once - see updateTurretAndTracers' exit-bounds despawn check, which the muzzle's intentionally off-screen spawn point would otherwise fail on frame 1. */
    hasEnteredBounds: boolean;
    /** Flicker frequencies/phases fixed at spawn, not re-rolled per frame - see TRACER_FLICKER_MAX_AMPLITUDE's doc comment. */
    flickerFrequency1Hz: number;
    flickerPhase1: number;
    flickerFrequency2Hz: number;
    flickerPhase2: number;
}

/** The single off-screen AA turret - see BattleScene.handlePointerDown/Move/Up and updateTurretAndTracers. */
interface TurretState {
    /** Toggled active for exactly the frame a round fires - see fireTracer/updateTurretAndTracers. */
    muzzleFlashObject: SceneObject;
    muzzleWorldX: number;
    muzzleWorldY: number;
    firing: boolean;
    aimWorldX: number;
    aimWorldY: number;
    /** Spread-ramp snapshot at the moment firing last toggled - see spreadFactorNow's doc comment. */
    spreadAtToggle: number;
    toggleTimeSeconds: number;
    nextRoundTimeSeconds: number;
}

/**
 * A Searchlight_N - lazily scans a slow sweep around its own authored resting angle
 * (scanCenterDegrees) until a 'flying' UFO crosses through its beam and a coin flip decides to lock
 * onto it (see the doc comment above SEARCHLIGHT_DETECTION_HALF_ANGLE_DEGREES), tracking it until
 * it's no longer 'flying' (destroyed or exited), then releases back to scanning until another
 * crossing. worldX/worldY are captured once at discovery (these lights never move, only rotate).
 * currentRotationDegrees is the beam's actual, physically-grounded aim - it only ever approaches
 * whatever the desired aim is (scanCenterDegrees's sweep, or the locked target's bearing) at up to
 * SEARCHLIGHT_MAX_TURN_RATE_DEGREES_PER_SECOND, so it never pops straight to a new angle.
 */
interface SearchlightInstance {
    root: SceneObject;
    worldX: number;
    worldY: number;
    scanCenterDegrees: number;
    scanAmplitudeDegrees: number;
    scanFrequencyHz: number;
    scanPhase0: number;
    lockedTarget: UfoInstance | null;
    /** 'flying' UFOs inside the beam's detection cone as of last frame - see ufosInBeam's use in updateSearchlights for why this is what makes a crossing a one-shot, entry-edge-triggered event rather than a per-frame coin flip. */
    ufosInBeam: Set<UfoInstance>;
    currentRotationDegrees: number;
}

/** A still-'flying' UFO's body world transform, cached once per frame (not once per bullet) - see BattleScene.computeFlyingUfoCollisionInfo. */
interface FlyingUfoCollisionInfo {
    ufo: UfoInstance;
    worldTransform: mat4;
    inverseWorldTransform: mat4;
    shape: PrimitiveShape;
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

    /** Slider-controlled multipliers - see onLoad's addSlider calls for range/step and where each is applied. */
    private laserBrightnessMultiplier = 1;
    private moonBrightnessMultiplier = 1;
    private ufoSpeedMultiplier = 1;
    private ufoFrequencyMultiplier = 1;
    private laserFrequencyMultiplier = 1;

    private ufos: UfoInstance[] = [];
    /** Elapsed wall-clock seconds since the scene loaded - the single "current time" every animation below is computed relative to, rather than each animation accumulating its own independent per-frame state. */
    private sceneTimeSeconds = 0;
    private nextSpawnTimeSeconds = randomRange(SPAWN_INTERVAL_MIN_SECONDS, SPAWN_INTERVAL_MAX_SECONDS);

    private clouds: CloudInstance[] = [];
    private hazeBands: HazeInstance[] = [];
    /** Cloud velocities need getSimulationBounds(), which isn't available yet during onLoad() (see the UFO spawn loop's own null-bounds guard) - set once, the first time onFrame() sees non-null bounds. */
    private cloudVelocitiesInitialized = false;

    private canvas!: HTMLCanvasElement;
    private tracerTemplate!: SceneObject;
    private turret!: TurretState;
    private tracers: TracerInstance[] = [];
    private searchlights: SearchlightInstance[] = [];

    /**
     * Pointer handlers as bound instance fields (not methods) so onUnload can remove the exact
     * same function reference it added in onLoad - the canvas is shared/persistent across scene
     * switches, so a stale, un-removed listener would otherwise outlive this scene instance.
     */
    private readonly handlePointerDown = (event: PointerEvent): void => {
        if (this.renderer.getActiveScene() !== this) {
            return;
        }
        this.canvas.setPointerCapture(event.pointerId);
        const world = this.renderer.screenToWorld(event.offsetX, event.offsetY);
        if (!world) {
            return;
        }
        this.turret.aimWorldX = world.x;
        this.turret.aimWorldY = world.y;
        this.turret.spreadAtToggle = this.spreadFactorNow(this.sceneTimeSeconds);
        this.turret.toggleTimeSeconds = this.sceneTimeSeconds;
        this.turret.firing = true;
        this.turret.nextRoundTimeSeconds = this.sceneTimeSeconds; // fire the first round immediately, not after a full interval
    };

    private readonly handlePointerMove = (event: PointerEvent): void => {
        if (this.renderer.getActiveScene() !== this || !this.turret.firing) {
            return;
        }
        const world = this.renderer.screenToWorld(event.offsetX, event.offsetY);
        if (!world) {
            return;
        }
        this.turret.aimWorldX = world.x;
        this.turret.aimWorldY = world.y;
    };

    private readonly handlePointerUp = (): void => {
        if (this.renderer.getActiveScene() !== this) {
            return;
        }
        this.turret.spreadAtToggle = this.spreadFactorNow(this.sceneTimeSeconds);
        this.turret.toggleTimeSeconds = this.sceneTimeSeconds;
        this.turret.firing = false;
    };

    public override onLoad(renderer: LitboxSceneRenderer): void {
        this.setActiveCamera('Main Camera');
        this.renderer = renderer;

        // The UFO object in the scene JSON is the template every spawned instance is cloned
        // from - deactivate it so the template itself is never rendered.
        this.ufoTemplate = this.getObject('UFO');
        this.ufoTemplate.active = false;

        const templateLaser = this.resolveRelativePath(this.ufoTemplate, 'Laser Gimbal/Laser');
        this.baseLaserIntensity = this.getLight(templateLaser).intensity;

        const moonSprite = this.getSprite(this.getObject('Moon'));
        const moonBaseColor = { ...moonSprite.colorMod };
        // The scene's one active static light ("Static Lights/Moonlight" - SpotLight and
        // DirectionalLight next to it are both authored inactive) - positioned directly above the
        // Moon sprite with a matching cool-white color, it's the moon's actual light source
        // (illuminating the haze/clouds), not the emissive sprite alone, so Moon Brightness scales
        // both together.
        const moonLight = this.getLight(this.getObject('Moonlight'));
        const moonLightBaseIntensity = moonLight.intensity;

        this.addSlider('Laser Brightness', 0, 3, 0.05,
            () => this.laserBrightnessMultiplier,
            (value) => { this.laserBrightnessMultiplier = value; });
        this.addSlider('Moon Brightness', 0, 3, 0.05,
            () => this.moonBrightnessMultiplier,
            (value) => {
                this.moonBrightnessMultiplier = value;
                moonSprite.colorMod = {
                    ...moonSprite.colorMod,
                    r: moonBaseColor.r * value,
                    g: moonBaseColor.g * value,
                    b: moonBaseColor.b * value,
                };
                this.markSpriteDirty(moonSprite);
                moonLight.intensity = moonLightBaseIntensity * value;
                this.markLightDirty(moonLight);
            });
        // UFO Speed/Frequency and Laser Frequency only take effect for UFOs spawned (or bursts
        // scheduled) after the slider moves - see spawnUfo/onFrame's spawn scheduling and
        // nextFireTimeSecondsFor, respectively - rather than retroactively rescaling in-flight
        // state, which would need to rebase each closed-form position/timing formula's origin to
        // avoid a visible jump.
        this.addSlider('UFO Speed', 0.1, 3, 0.05,
            () => this.ufoSpeedMultiplier,
            (value) => { this.ufoSpeedMultiplier = value; });
        this.addSlider('UFO Frequency', 0.1, 3, 0.05,
            () => this.ufoFrequencyMultiplier,
            (value) => { this.ufoFrequencyMultiplier = value; });
        this.addSlider('Laser Frequency', 0.1, 3, 0.05,
            () => this.laserFrequencyMultiplier,
            (value) => { this.laserFrequencyMultiplier = value; });

        // Cloud_N/Haze_L_N/Haze_R_N/Searchlight_N counts are all arbitrary (not assumed to match
        // whatever's currently authored in battle.json), so all are discovered by name pattern
        // rather than indexed by number - see each interface's doc comment for why velocityX's
        // initial value differs between Cloud_N and Haze_L_N/Haze_R_N (haze's is fixed here;
        // cloud's needs simulation bounds, not yet available this early - see
        // cloudVelocitiesInitialized).
        for (const obj of this.data.objects) {
            if (/^Cloud_\d+$/.test(obj.name)) {
                const localBounds = this.computeLocalBounds(obj);
                const worldPosition = this.computeWorldPosition(obj);
                this.clouds.push({ root: obj, localBounds, worldX: worldPosition.x, worldY: worldPosition.y, velocityX: 0 });
            } else if (/^Haze_L_\d+$/.test(obj.name)) {
                const localBounds = this.computeLocalBounds(obj);
                const worldPosition = this.computeWorldPosition(obj);
                this.hazeBands.push({ root: obj, localBounds, worldX: worldPosition.x, worldY: worldPosition.y, velocityX: -HAZE_SPEED });
            } else if (/^Haze_R_\d+$/.test(obj.name)) {
                const localBounds = this.computeLocalBounds(obj);
                const worldPosition = this.computeWorldPosition(obj);
                this.hazeBands.push({ root: obj, localBounds, worldX: worldPosition.x, worldY: worldPosition.y, velocityX: HAZE_SPEED });
            } else if (/^Searchlight_\d+$/.test(obj.name)) {
                const worldPosition = this.computeWorldPosition(obj);
                // 'dynamic', not per-frame markTransformDirty calls - a searchlight's rotation
                // genuinely changes every frame for the rest of the scene's life (scanning or
                // locked-on), which is exactly what makeTransformDynamic is for. See
                // updateSearchlights, which no longer calls markTransformDirty for this reason.
                this.makeTransformDynamic(obj);
                this.searchlights.push({
                    root: obj,
                    worldX: worldPosition.x,
                    worldY: worldPosition.y,
                    // The authored rotation is this searchlight's intended resting/center aim, not a placeholder 0.
                    scanCenterDegrees: obj.rotation,
                    scanAmplitudeDegrees: randomRange(SEARCHLIGHT_SCAN_AMPLITUDE_MIN_DEGREES, SEARCHLIGHT_SCAN_AMPLITUDE_MAX_DEGREES),
                    scanFrequencyHz: randomRange(SEARCHLIGHT_SCAN_FREQUENCY_MIN_HZ, SEARCHLIGHT_SCAN_FREQUENCY_MAX_HZ),
                    scanPhase0: Math.random() * Math.PI * 2,
                    lockedTarget: null,
                    ufosInBeam: new Set(),
                    currentRotationDegrees: obj.rotation,
                });
            }
        }

        // The Tracer object in the scene JSON is the template every fired round is cloned from -
        // deactivate it so the template itself is never rendered (mirrors ufoTemplate above).
        this.tracerTemplate = this.getObject('Tracer');
        this.tracerTemplate.active = false;

        const muzzleFlashObject = this.getObject('MuzzleFlash');
        muzzleFlashObject.active = false; // resting state is off - see updateTurretAndTracers' single-frame strobe
        const muzzleWorldPosition = this.computeWorldPosition(muzzleFlashObject);
        this.turret = {
            muzzleFlashObject,
            muzzleWorldX: muzzleWorldPosition.x,
            muzzleWorldY: muzzleWorldPosition.y,
            firing: false,
            aimWorldX: muzzleWorldPosition.x,
            aimWorldY: muzzleWorldPosition.y + 1,
            spreadAtToggle: 0,
            toggleTimeSeconds: 0,
            nextRoundTimeSeconds: 0,
        };

        this.canvas = renderer.getCanvas();
        this.canvas.style.touchAction = 'none'; // a firing drag on mobile shouldn't also scroll/zoom the page
        this.canvas.addEventListener('pointerdown', this.handlePointerDown);
        this.canvas.addEventListener('pointermove', this.handlePointerMove);
        this.canvas.addEventListener('pointerup', this.handlePointerUp);
        this.canvas.addEventListener('pointercancel', this.handlePointerUp);
    }

    public override onUnload(): void {
        this.canvas.removeEventListener('pointerdown', this.handlePointerDown);
        this.canvas.removeEventListener('pointermove', this.handlePointerMove);
        this.canvas.removeEventListener('pointerup', this.handlePointerUp);
        this.canvas.removeEventListener('pointercancel', this.handlePointerUp);
    }

    public override getDenoiserTunables(): Partial<DenoiserTunables> {
        return DENOISER_TUNABLES;
    }

    public override getStatusText(): string {
        return 'UFO battle - click/tap and hold to fire the AA turret';
    }

    public override onFrame(deltaTimeSeconds: number): void {
        this.sceneTimeSeconds += deltaTimeSeconds;

        const bounds = this.getSimulationBounds();
        if (!bounds) {
            return;
        }

        if (this.sceneTimeSeconds >= this.nextSpawnTimeSeconds) {
            this.nextSpawnTimeSeconds =
                this.sceneTimeSeconds +
                randomRange(SPAWN_INTERVAL_MIN_SECONDS, SPAWN_INTERVAL_MAX_SECONDS) / this.ufoFrequencyMultiplier;
            this.spawnUfo(bounds);
        }

        for (const ufo of this.ufos) {
            this.updateUfo(ufo, this.sceneTimeSeconds, bounds);
        }
        this.despawnExited(bounds);

        this.updateTurretAndTracers(this.sceneTimeSeconds, deltaTimeSeconds, bounds);
        this.updateSearchlights(this.sceneTimeSeconds, deltaTimeSeconds);

        if (!this.cloudVelocitiesInitialized) {
            for (const cloud of this.clouds) {
                cloud.velocityX = -this.cloudSpeedForHeightFraction(this.heightFractionFor(cloud.worldY, bounds));
            }
            this.cloudVelocitiesInitialized = true;
        }
        this.updateClouds(bounds, deltaTimeSeconds);
        this.updateHaze(bounds, deltaTimeSeconds);
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
        const horizontalSpeed = (areaWidth / crossingSeconds) * SPEED_MULTIPLIER * this.ufoSpeedMultiplier;
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

        const bodyObject = this.resolveRelativePath(root, 'Bottom/Traced');
        const bodySprite = this.getSprite(this.resolveRelativePath(root, 'Bottom/Sprite'));
        const bodyRaytraced = this.getRaytraced(bodyObject);
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
            bodyObject,
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

    /** Per-UFO next-fire time, normalized so the combined firing rate stays population-invariant (times FIRE_FREQUENCY_MULTIPLIER, then the Laser Frequency slider). */
    private nextFireTimeSecondsFor(population: number): number {
        return (
            this.sceneTimeSeconds +
            (randomRange(FIRE_INTERVAL_MIN_SECONDS, FIRE_INTERVAL_MAX_SECONDS) * population) /
                (FIRE_FREQUENCY_MULTIPLIER * this.laserFrequencyMultiplier)
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
            // Oscillates between baseLaserIntensity/2 (low) and baseLaserIntensity (high, the template's own
            // value), both scaled by the Laser Brightness slider.
            const oscillation = 0.5 + 0.5 * Math.sin(2 * Math.PI * LASER_FLICKER_FREQUENCY_HZ * fireElapsed);
            ufo.laserLight.intensity = ((this.baseLaserIntensity * this.laserBrightnessMultiplier) / 2) * (1 + oscillation);
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

    /** Cloud_N's height, as a 0 (bounds.bottom) to 1 (bounds.top) fraction - the single input both the initial and every respawned velocityX are derived from (see cloudSpeedForHeightFraction). */
    private heightFractionFor(y: number, bounds: SimulationBounds): number {
        return (y - bounds.bottom) / (bounds.top - bounds.bottom);
    }

    /**
     * Higher clouds drift slower - a linear interpolation between CLOUD_SPEED_AT_LOW_ALTITUDE and
     * CLOUD_SPEED_AT_HIGH_ALTITUDE across the [CLOUD_HEIGHT_FRACTION_MIN, MAX] range, clamped so a
     * cloud already outside that range (e.g. wherever battle.json happened to author one) still
     * gets a sane speed instead of an extrapolated one.
     */
    private cloudSpeedForHeightFraction(heightFraction: number): number {
        const normalized = Math.min(
            Math.max((heightFraction - CLOUD_HEIGHT_FRACTION_MIN) / (CLOUD_HEIGHT_FRACTION_MAX - CLOUD_HEIGHT_FRACTION_MIN), 0),
            1,
        );
        return CLOUD_SPEED_AT_LOW_ALTITUDE + (CLOUD_SPEED_AT_HIGH_ALTITUDE - CLOUD_SPEED_AT_LOW_ALTITUDE) * normalized;
    }

    /**
     * Drifts every Cloud_N left in world space (see CloudInstance's doc comment for why worldX,
     * not root.position.x, is what's tracked), respawning (see respawnCloud) whichever ones have
     * fully exited to the left, then pushes the result into root.position via setWorldPosition.
     */
    private updateClouds(bounds: SimulationBounds, deltaTimeSeconds: number): void {
        for (const cloud of this.clouds) {
            cloud.worldX += cloud.velocityX * deltaTimeSeconds;

            const rightEdge = cloud.worldX + cloud.localBounds.maxX;
            if (rightEdge < bounds.left) {
                this.respawnCloud(cloud, bounds);
            }

            this.setWorldPosition(cloud.root, cloud.worldX, cloud.worldY);
            this.markTransformDirty(cloud.root);
        }
    }

    /** Re-randomizes a cloud's height (within CLOUD_HEIGHT_FRACTION_MIN/MAX) and matching speed, and places it just off the right edge of the simulation area (in world space). */
    private respawnCloud(cloud: CloudInstance, bounds: SimulationBounds): void {
        const heightFraction = randomRange(CLOUD_HEIGHT_FRACTION_MIN, CLOUD_HEIGHT_FRACTION_MAX);
        cloud.worldY = bounds.bottom + heightFraction * (bounds.top - bounds.bottom);
        cloud.velocityX = -this.cloudSpeedForHeightFraction(heightFraction);
        cloud.worldX = bounds.right - cloud.localBounds.minX;
    }

    /**
     * Drifts every Haze_L_N/Haze_R_N at its own constant, fixed-sign velocityX (in world space -
     * see CloudInstance's doc comment), wrapping each one to just off the opposite edge once it's
     * fully exited - see HazeInstance.
     */
    private updateHaze(bounds: SimulationBounds, deltaTimeSeconds: number): void {
        for (const haze of this.hazeBands) {
            haze.worldX += haze.velocityX * deltaTimeSeconds;

            if (haze.velocityX < 0) {
                const rightEdge = haze.worldX + haze.localBounds.maxX;
                if (rightEdge < bounds.left) {
                    haze.worldX = bounds.right - haze.localBounds.minX;
                }
            } else {
                const leftEdge = haze.worldX + haze.localBounds.minX;
                if (leftEdge > bounds.right) {
                    haze.worldX = bounds.left - haze.localBounds.maxX;
                }
            }

            this.setWorldPosition(haze.root, haze.worldX, haze.worldY);
            this.markTransformDirty(haze.root);
        }
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
        const angleToTargetDegrees = this.angleToDegrees(gimbalWorldX, gimbalWorldY, target.root.position.x, target.root.position.y);
        // The Laser's own rotation (fixed, relative to the gimbal) is the template's baked beam
        // offset - the gimbal's rotation is the only aim adjustment we make. forward_monte_carlo.wgsl's
        // LIGHT_KIND_LASER emitter fires along the Laser object's local -Y axis (not +X - its scale,
        // 0.05 wide by 0.142 tall, is a tall thin beam shape), which points at world angle
        // (totalRotation + 90deg); solving totalRotation = gimbal.rotation + laser.rotation for the
        // beam to point at the target adds that same 90deg here.
        shooter.gimbal.rotation = angleToTargetDegrees + 90 - shooter.laser.rotation;
        this.markTransformDirty(shooter.gimbal);
    }

    /** World-space angle (degrees, matching Math.atan2's own convention) from (fromX,fromY) to (toX,toY) - shared by aimGimbalAt and the searchlight lock-on aim below. */
    private angleToDegrees(fromX: number, fromY: number, toX: number, toY: number): number {
        return (Math.atan2(toY - fromY, toX - fromX) * 180) / Math.PI;
    }

    /**
     * Shortest signed difference (-180,180] from `fromDegrees` to `toDegrees` - JS `%` keeps the
     * sign of its left operand, so a single modulo only guarantees a result in (-360,360); the
     * second adjustment folds that into the shortest-path range. Shared by stepAngleTowards (how
     * far a searchlight beam still needs to turn) and updateSearchlights' detection-cone check (how
     * far off-boresight a candidate UFO currently is).
     */
    private angleDifferenceDegrees(fromDegrees: number, toDegrees: number): number {
        let delta = (toDegrees - fromDegrees) % 360;
        if (delta > 180) {
            delta -= 360;
        } else if (delta < -180) {
            delta += 360;
        }
        return delta;
    }

    /**
     * Steps `currentDegrees` toward `targetDegrees` by at most `maxDeltaDegrees`, always turning
     * the short way around the +-180 degree wraparound seam - the physically-grounded alternative
     * to just assigning `targetDegrees` outright (which is what let the searchlights "pop" straight
     * to a new aim). Used by updateSearchlights for both the scan sweep and lock-on tracking, so
     * neither can ever produce an instantaneous jump.
     */
    private stepAngleTowards(currentDegrees: number, targetDegrees: number, maxDeltaDegrees: number): number {
        const delta = this.angleDifferenceDegrees(currentDegrees, targetDegrees);
        const clampedDelta = Math.max(-maxDeltaDegrees, Math.min(maxDeltaDegrees, delta));
        return currentDegrees + clampedDelta;
    }

    /**
     * The turret's current accuracy spread, 0 (perfectly aimed) to 1 (max jitter) - a closed-form
     * formula relative to a snapshot taken at the moment firing last toggled (turret.spreadAtToggle/
     * toggleTimeSeconds, set in handlePointerDown/Up), not an accumulated per-frame delta. This is
     * what makes the very first round of a fresh trigger-pull always land at spread=0 with no
     * special-casing (elapsed=0 at that instant), and makes resuming fire before a previous burst's
     * spread fully decayed correctly resume from wherever it left off rather than resetting.
     */
    private spreadFactorNow(nowSeconds: number): number {
        const elapsed = nowSeconds - this.turret.toggleTimeSeconds;
        return this.turret.firing
            ? Math.min(1, this.turret.spreadAtToggle + elapsed / TURRET_SPREAD_RAMP_UP_SECONDS)
            : Math.max(0, this.turret.spreadAtToggle - elapsed / TURRET_SPREAD_RAMP_DOWN_SECONDS);
    }

    /** Turret firing/muzzle-flash/tracer-lifecycle update, called once per frame regardless of whether the turret is currently firing (tracers already in flight still need updating/despawning). */
    private updateTurretAndTracers(nowSeconds: number, deltaTimeSeconds: number, bounds: SimulationBounds): void {
        if (this.turret.muzzleFlashObject.active) {
            this.turret.muzzleFlashObject.active = false;
            this.markTransformDirty(this.turret.muzzleFlashObject);
        }

        // A while loop (not if) so a slow/hitched frame catches up on rounds due, rather than
        // silently dropping them - matches this file's general "derive from real elapsed time"
        // philosophy rather than assuming onFrame runs at a fixed rate.
        while (this.turret.firing && nowSeconds >= this.turret.nextRoundTimeSeconds) {
            this.fireTracer(this.turret.nextRoundTimeSeconds);
            this.turret.nextRoundTimeSeconds += 1 / TURRET_FIRE_RATE_HZ;
        }

        if (this.tracers.length === 0) {
            return;
        }

        const flyingUfos = this.computeFlyingUfoCollisionInfo();
        const survivors: TracerInstance[] = [];
        for (const tracer of this.tracers) {
            if (nowSeconds - tracer.spawnTimeSeconds >= TRACER_LIFETIME_SECONDS) {
                // Burned out - despawns outright, no collision/bounds test needed.
                this.destroyObject(tracer.root);
                continue;
            }

            // Collision is tested against the segment from last frame's position to this frame's -
            // both evaluated from the same closed-form formula, so no extra "previous position"
            // field needs storing - this is what keeps a fast bullet from tunneling through a thin
            // UFO silhouette between two frames. Clamped to spawnTimeSeconds so a tracer's very
            // first frame doesn't test a phantom pre-spawn segment.
            const prevPos = this.tracerPositionAt(tracer, Math.max(nowSeconds - deltaTimeSeconds, tracer.spawnTimeSeconds));
            const currPos = this.tracerPositionAt(tracer, nowSeconds);

            const hitUfo = this.findTracerHit(prevPos, currPos, flyingUfos);
            if (hitUfo) {
                // A bullet is always consumed on impact - only whether it downs the target is a coin flip.
                this.destroyObject(tracer.root);
                if (Math.random() < TRACER_HIT_DOWN_CHANCE) {
                    this.applyHit(hitUfo, nowSeconds);
                }
                continue;
            }

            const withinBounds =
                currPos.x >= bounds.left && currPos.x <= bounds.right && currPos.y >= bounds.bottom && currPos.y <= bounds.top;
            if (withinBounds) {
                tracer.hasEnteredBounds = true;
            } else if (tracer.hasEnteredBounds) {
                // Despawns immediately on exit, unlike UFOs - no OFFSCREEN_MARGIN. Gated on
                // hasEnteredBounds because the muzzle's spawn point is deliberately below
                // bounds.bottom (off-screen) - without this gate, every tracer would fail this
                // check (and be destroyed) on the very first frame, before ever having risen into
                // view - "exiting" only means something for a tracer that was actually inside.
                this.destroyObject(tracer.root);
                continue;
            }

            this.updateTracerTransform(tracer, currPos, nowSeconds);
            this.updateTracerBrightness(tracer, nowSeconds);
            survivors.push(tracer);
        }
        this.tracers = survivors;
    }

    /** Fires one round at `atSeconds` (the round's scheduled time, not necessarily nowSeconds - see the catch-up while loop in updateTurretAndTracers). */
    private fireTracer(atSeconds: number): void {
        this.turret.muzzleFlashObject.active = true;
        this.markTransformDirty(this.turret.muzzleFlashObject);

        const spread = this.spreadFactorNow(atSeconds);
        const aimAngleDegrees = this.angleToDegrees(
            this.turret.muzzleWorldX, this.turret.muzzleWorldY, this.turret.aimWorldX, this.turret.aimWorldY,
        );
        const jitteredAngleRadians = ((aimAngleDegrees + randomRange(-1, 1) * TURRET_SPREAD_MAX_DEGREES * spread) * Math.PI) / 180;

        const speed = TRACER_SPEED * randomRange(1 - TRACER_SPEED_JITTER_FRACTION, 1 + TRACER_SPEED_JITTER_FRACTION);
        const velocityX = speed * Math.cos(jitteredAngleRadians);
        const velocityY = speed * Math.sin(jitteredAngleRadians);

        // No position override here - Tracer's parent (Scene) carries its own nonzero world
        // offset, so a world-space coordinate can't be assigned to cloneObject's (parent-relative)
        // position option directly (see setWorldPosition's doc comment). The tracer-update loop in
        // updateTurretAndTracers calls updateTracerTransform on this same object this same frame,
        // which uses setWorldPosition to place it correctly before anything is ever rendered.
        const root = this.cloneObject(this.tracerTemplate, { active: true });
        // 'dynamic' (re-uploaded every frame for this object's whole life), not repeated
        // markTransformDirty calls - a tracer's transform genuinely changes every frame it exists,
        // which is exactly what makeTransformDynamic (not the one-off 'dirty' flag) is for. See
        // updateTracerTransform, which no longer calls markTransformDirty for this reason.
        this.makeTransformDynamic(root);

        // Brightness (see updateTracerBrightness) also changes every frame for this object's whole
        // life, so its sprite/light get the same 'dynamic' treatment as the transform above.
        const sprite = this.getSprite(this.resolveRelativePath(root, 'Sprite'));
        const light = this.getLight(this.resolveRelativePath(root, 'PointLight'));
        this.makeSpriteDynamic(sprite);
        this.makeLightDynamic(light);

        this.tracers.push({
            root,
            sprite,
            light,
            baseEmissive: { ...sprite.emissive },
            baseLightIntensity: light.intensity,
            spawnTimeSeconds: atSeconds,
            spawnX: this.turret.muzzleWorldX,
            spawnY: this.turret.muzzleWorldY,
            velocityX,
            velocityYAtSpawn: velocityY,
            hasEnteredBounds: false,
            flickerFrequency1Hz: randomRange(TRACER_FLICKER_FREQUENCY_MIN_HZ, TRACER_FLICKER_FREQUENCY_MAX_HZ),
            flickerPhase1: Math.random() * Math.PI * 2,
            flickerFrequency2Hz: randomRange(TRACER_FLICKER_FREQUENCY_MIN_HZ, TRACER_FLICKER_FREQUENCY_MAX_HZ),
            flickerPhase2: Math.random() * Math.PI * 2,
        });
    }

    /** Closed-form position at `atSeconds` - gravity only touches the Y term, so this stays a pure function of elapsed time, no accumulation. */
    private tracerPositionAt(tracer: TracerInstance, atSeconds: number): Vector2 {
        const elapsed = atSeconds - tracer.spawnTimeSeconds;
        return {
            x: tracer.spawnX + tracer.velocityX * elapsed,
            y: tracer.spawnY + tracer.velocityYAtSpawn * elapsed - 0.5 * TRACER_GRAVITY * elapsed * elapsed,
        };
    }

    /** Position/rotation/stretch for a tracer that survived this frame's collision/despawn checks - all derived from elapsed time, matching updateFalling's style. */
    private updateTracerTransform(tracer: TracerInstance, currPos: Vector2, nowSeconds: number): void {
        const elapsed = nowSeconds - tracer.spawnTimeSeconds;
        const velocityY = tracer.velocityYAtSpawn - TRACER_GRAVITY * elapsed;
        const speed = Math.hypot(tracer.velocityX, velocityY);

        // World-space currPos can't be assigned to tracer.root.position directly - Tracer's parent
        // (Scene) carries its own nonzero offset, so .position (parent-relative) and world space
        // aren't the same numbers here. See setWorldPosition's doc comment (same reasoning
        // updateClouds/updateHaze already rely on for exactly this scene).
        this.setWorldPosition(tracer.root, currPos.x, currPos.y);
        // Local +Y is the Tracer template's stretch axis (at rotation=0, litbox_scene.ts's
        // mat4.rotateZ convention maps local +Y to world (0,1) - the opposite sign from the UFO
        // laser's local -Y forward axis, so this is the same family of angle+offset formula
        // aimGimbalAt/angleToDegrees uses, just with the sign flipped: atan2(vy,vx) - 90deg instead
        // of + 90deg).
        tracer.root.rotation = (Math.atan2(velocityY, tracer.velocityX) * 180) / Math.PI - 90;
        tracer.root.scale.y = 1 + speed * TRACER_STRETCH_PER_SPEED;
        // No markTransformDirty call here - fireTracer already made this object's transform
        // 'dynamic' once at spawn, which alone is what makes the renderer re-derive/re-upload it
        // every frame from here on (see DynamicSet.activeThisFrame/LitboxSceneRenderer.
        // applyDynamicSceneUpdates); re-marking 'dirty' every frame on top of that would be a no-op
        // (DynamicSet.markDirty is a guarded no-op once something is already 'dynamic').
    }

    /**
     * Fading + increasingly erratic flicker as a tracer burns out - see TRACER_LIFETIME_SECONDS
     * and TRACER_FLICKER_MAX_AMPLITUDE's doc comments. Scales baseEmissive/baseLightIntensity
     * (captured once at spawn) rather than the sprite/light's current values, so repeated frames
     * don't compound the scaling. No markSpriteDirty/markLightDirty call needed - fireTracer
     * already made both 'dynamic', same reasoning as updateTracerTransform's own comment.
     */
    private updateTracerBrightness(tracer: TracerInstance, nowSeconds: number): void {
        const elapsed = nowSeconds - tracer.spawnTimeSeconds;
        const ageFraction = Math.min(elapsed / TRACER_LIFETIME_SECONDS, 1);
        const envelope = 1 - ageFraction; // steady fuel-burn fade toward 0

        const flickerAmplitude = TRACER_FLICKER_MAX_AMPLITUDE * ageFraction;
        const noise =
            0.6 * Math.sin(2 * Math.PI * tracer.flickerFrequency1Hz * elapsed + tracer.flickerPhase1) +
            0.4 * Math.sin(2 * Math.PI * tracer.flickerFrequency2Hz * elapsed + tracer.flickerPhase2);
        const brightness = Math.max(0, envelope * (1 + flickerAmplitude * noise));

        tracer.sprite.emissive = {
            ...tracer.sprite.emissive,
            r: tracer.baseEmissive.r * brightness,
            g: tracer.baseEmissive.g * brightness,
            b: tracer.baseEmissive.b * brightness,
        };
        tracer.light.intensity = tracer.baseLightIntensity * brightness;
    }

    /** Caches every still-'flying' UFO's body world transform once per frame - computeWorldTransform is documented as not meant for a per-frame-per-object hot path, so this keeps tracer collision at O(UFOs) instead of O(bullets x UFOs). */
    private computeFlyingUfoCollisionInfo(): FlyingUfoCollisionInfo[] {
        const infos: FlyingUfoCollisionInfo[] = [];
        for (const ufo of this.ufos) {
            if (ufo.state !== 'flying') {
                continue;
            }
            const worldTransform = this.computeWorldTransform(ufo.bodyObject);
            const inverseWorldTransform = mat4.create();
            if (!mat4.invert(inverseWorldTransform, worldTransform)) {
                continue; // not invertible (e.g. a zero scale somewhere in its ancestry) - skip collision for this UFO this frame
            }
            const shape: PrimitiveShape = ufo.bodyRaytraced.primitiveShape === 'rect' ? 'rect' : 'ellipse';
            infos.push({ ufo, worldTransform, inverseWorldTransform, shape });
        }
        return infos;
    }

    private findTracerHit(prevPos: Vector2, currPos: Vector2, flyingUfos: FlyingUfoCollisionInfo[]): UfoInstance | null {
        for (const info of flyingUfos) {
            if (testSegmentAgainstShape(prevPos, currPos, info.worldTransform, info.inverseWorldTransform, info.shape)) {
                return info.ufo;
            }
        }
        return null;
    }

    /**
     * Scan/lock-on update for every Searchlight_N, independent per instance. Once locked, a light
     * keeps tracking its target for as long as it stays a valid 'flying' UFO; a lock only releases
     * when the target is gone (destroyed or exited), back to lazily scanning. While scanning, the
     * light doesn't go looking for a target scene-wide - it only notices a 'flying' UFO that's
     * currently within its beam's own detection cone (SEARCHLIGHT_DETECTION_HALF_ANGLE_DEGREES of
     * the beam's actual current aim, currentRotationDegrees), and only on the frame that UFO enters
     * the cone (tracked via ufosInBeam, so lingering inside it doesn't re-roll every frame) - each
     * such crossing is an independent SEARCHLIGHT_ACQUIRE_CHANCE coin flip for whether the light
     * bothers to lock on. Either way, the light's actual rotation never jumps straight to the
     * desired aim - it's turned toward it at a bounded rate via stepAngleTowards, so switching
     * between scanning and tracking, or between one target and the next, is always a
     * physically-grounded swing rather than a pop.
     */
    private updateSearchlights(nowSeconds: number, deltaTimeSeconds: number): void {
        for (const light of this.searchlights) {
            const targetInvalid =
                light.lockedTarget !== null &&
                (!this.ufos.includes(light.lockedTarget) || light.lockedTarget.state !== 'flying');
            if (light.lockedTarget && targetInvalid) {
                light.lockedTarget = null;
                // Cleared, not carried forward - re-acquiring after a release needs a fresh
                // crossing, not credit for whatever already happened to be sitting in the cone.
                light.ufosInBeam.clear();
            }

            if (!light.lockedTarget) {
                const inBeamNow = new Set<UfoInstance>();
                for (const ufo of this.ufos) {
                    if (ufo.state !== 'flying') {
                        continue;
                    }
                    const bearingDegrees = this.angleToDegrees(light.worldX, light.worldY, ufo.root.position.x, ufo.root.position.y) + 90;
                    const offBoresightDegrees = Math.abs(this.angleDifferenceDegrees(light.currentRotationDegrees, bearingDegrees));
                    if (offBoresightDegrees > SEARCHLIGHT_DETECTION_HALF_ANGLE_DEGREES) {
                        continue;
                    }
                    inBeamNow.add(ufo);
                    if (!light.ufosInBeam.has(ufo) && Math.random() < SEARCHLIGHT_ACQUIRE_CHANCE) {
                        light.lockedTarget = ufo;
                        break;
                    }
                }
                light.ufosInBeam = inBeamNow;
            }

            const desiredRotationDegrees = light.lockedTarget
                ? this.angleToDegrees(light.worldX, light.worldY, light.lockedTarget.root.position.x, light.lockedTarget.root.position.y) + 90
                : light.scanCenterDegrees + light.scanAmplitudeDegrees * Math.sin(2 * Math.PI * light.scanFrequencyHz * nowSeconds + light.scanPhase0);
            light.currentRotationDegrees = this.stepAngleTowards(
                light.currentRotationDegrees, desiredRotationDegrees, SEARCHLIGHT_MAX_TURN_RATE_DEGREES_PER_SECOND * deltaTimeSeconds);
            light.root.rotation = light.currentRotationDegrees;
            // No markTransformDirty call - see the makeTransformDynamic call at discovery time.
        }
    }
}
