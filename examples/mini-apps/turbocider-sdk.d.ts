export type Permission = "media.import" | "media.export" | "image.generate" | "adapters.use";
export type Engine = "qwen-image-2.1-turbo" | "qwen-image-2.1" | "flux2-klein-4b" | "z-image-turbo";
export type Role = "person" | "garment" | "pose" | "depth" | "expression" | "hairstyle"
  | "scene" | "lighting" | "prop" | "furniture" | "reference";
export type Reference = { role: Role } & ({ mediaID: string; shotIndex?: never } | { shotIndex: number; mediaID?: never });

export interface Media {
  id: string;
  url: string;
  name: string;
  width: number;
  height: number;
}

export interface Shot {
  prompt: string;
  references: Reference[];
  width?: number;
  height?: number;
  matchReferenceAspect?: boolean;
  referencePixelBudget?: number;
  seed?: number;
  adapters?: { id: string; strength?: number }[];
  label?: string;
  context?: string;
}

export interface PlanOptions {
  requestID: string;
  engine?: Engine;
  workflow: string;
  shots: Shot[];
}

export interface Plan {
  id: string;
  jobs: {
    index: number;
    seed: number;
    steps: number;
    width: number;
    height: number;
    referenceCount: number;
    referencePixelBudget?: number;
    residency?: "streamed" | "component_staged";
    memory: Partial<{ estimatedBytes: number; reserveBytes: number; physicalBytes: number }>;
  }[];
}

export interface Job {
  id: string;
  requestID: string;
  workflow: string;
  state: "queued" | "preparing" | "running" | "cancelling" | "finalizing" | "cleanup_pending"
    | "succeeded" | "failed" | "cancelled" | "interrupted";
  phase: string;
  completed: number;
  total: number;
  label: string;
  seed: number;
  model: string;
  createdAt: number;
  prompt: string;
  referenceIDs: string[];
  referencePixelBudget?: number;
  referencePixels?: number;
  dependencies: string[];
  context?: string;
  output?: Media;
  error?: string;
}

export interface MiniAppHost {
  readonly sdkVersion: 1;
  createRequestID(): string;
  call(method: "handshake"): Promise<{
    sdkVersion: 1;
    appID: string;
    appVersion: string;
    permissions: Permission[];
    maxReferences: number;
    maxBatchSize: number;
    supportsDependentShots: boolean;
  }>;
  call(method: "models.list"): Promise<{ id: Engine; installed: boolean }[]>;
  call(method: "adapters.list", payload: { engine: Engine }): Promise<{ id: string; name: string; strength: number }[]>;
  readonly media: {
    bundled(path: string): Promise<Media>;
    import(blob: Blob, name?: string): Promise<Media>;
    get(id: string): Promise<Media>;
    save(id: string): Promise<boolean>;
    saveBlob(blob: Blob, name: string): Promise<boolean>;
  };
  readonly storage: {
    get(key: string): Promise<string | null>;
    set(key: string, value: string | null): Promise<boolean>;
  };
  readonly workflow: { plan(options: PlanOptions): Promise<Plan> };
  readonly jobs: {
    submit(planID: string): Promise<Job[]>;
    get(id: string): Promise<Job>;
    watch(id: string, options?: { signal?: AbortSignal }): AsyncIterable<Job>;
    list(): Promise<Job[]>;
    cancel(id: string): Promise<boolean>;
    remove(id: string): Promise<boolean>;
  };
}

declare global {
  interface Window { readonly TurboCider: MiniAppHost }
}
