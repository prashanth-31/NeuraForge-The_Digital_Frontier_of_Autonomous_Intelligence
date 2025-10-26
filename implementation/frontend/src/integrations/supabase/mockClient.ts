import { AuthError, AuthChangeEvent, Session, User } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";

type AuthListener = (event: AuthChangeEvent, session: Session | null) => void;

type StoredUser = {
  id: string;
  email: string;
  password: string;
  name: string;
  createdAt: string;
  updatedAt: string;
};

type StoredSession = {
  userId: string;
  expiresAt: number;
};

const USERS_KEY = "mock-supabase-users";
const SESSION_KEY = "mock-supabase-session";

const listeners = new Set<AuthListener>();

const storage = typeof window !== "undefined" ? window.localStorage : undefined;

const DEFAULT_USERS: Record<string, StoredUser> = {
  "demo-user": {
    id: "demo-user",
    email: "demo@neuraforge.ai",
    password: "NeuraForge123!",
    name: "Demo Explorer",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
};

const memoryStore: { users: Record<string, StoredUser>; session: StoredSession | null } = {
  users: {},
  session: null,
};

function uuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2, 10);
}

function loadUsers(): Record<string, StoredUser> {
  if (!storage) {
    if (Object.keys(memoryStore.users).length === 0) {
      memoryStore.users = { ...DEFAULT_USERS };
    }
    return memoryStore.users;
  }
  const raw = storage.getItem(USERS_KEY);
  if (!raw) {
    return { ...DEFAULT_USERS };
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, StoredUser>;
    return Object.keys(parsed).length === 0 ? { ...DEFAULT_USERS } : parsed;
  } catch {
    return { ...DEFAULT_USERS };
  }
}

function saveUsers(users: Record<string, StoredUser>): void {
  if (!storage) {
    memoryStore.users = users;
    return;
  }
  storage.setItem(USERS_KEY, JSON.stringify(users));
}

function loadStoredSession(): StoredSession | null {
  if (!storage) {
    return memoryStore.session;
  }
  const raw = storage.getItem(SESSION_KEY);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw) as StoredSession;
  } catch {
    return null;
  }
}

function saveStoredSession(session: StoredSession | null): void {
  if (!storage) {
    memoryStore.session = session;
    return;
  }
  if (session) {
    storage.setItem(SESSION_KEY, JSON.stringify(session));
  } else {
    storage.removeItem(SESSION_KEY);
  }
}

function buildUser(stored: StoredUser): User {
  const now = new Date().toISOString();
  return {
    id: stored.id,
    aud: "authenticated",
    role: "authenticated",
    email: stored.email,
    phone: "",
    app_metadata: {
      provider: "email",
      providers: ["email"],
    },
    user_metadata: {
      full_name: stored.name,
    },
    identities: [],
    created_at: stored.createdAt,
    updated_at: stored.updatedAt,
    last_sign_in_at: now,
    confirmed_at: stored.createdAt,
    email_confirmed_at: stored.createdAt,
    raw_user_meta_data: {
      full_name: stored.name,
    },
    raw_app_meta_data: {
      provider: "email",
      providers: ["email"],
    },
    is_anonymous: false,
    factors: [],
    recovery_sent_at: null,
    new_email: null,
    new_phone: null,
    phone_confirmed_at: null,
    phone_change_sent_at: null,
    phone_change_token: null,
    phone_change: null,
    email_change: null,
    email_change_sent_at: null,
    email_change_token: null,
    invited_at: null,
  } as User;
}

function buildSession(stored: StoredUser): Session {
  const user = buildUser(stored);
  const expiresIn = 60 * 60;
  const expiresAt = Math.floor(Date.now() / 1000) + expiresIn;
  return {
    access_token: uuid(),
    token_type: "bearer",
    expires_in: expiresIn,
    expires_at: expiresAt,
    refresh_token: uuid(),
    user,
    provider_token: null,
    provider_refresh_token: null,
  } as Session;
}

function getActiveSession(): Session | null {
  const storedSession = loadStoredSession();
  if (!storedSession) {
    return null;
  }
  const users = loadUsers();
  const storedUser = users[storedSession.userId];
  if (!storedUser) {
    saveStoredSession(null);
    return null;
  }
  if (storedSession.expiresAt <= Date.now()) {
    saveStoredSession(null);
    return null;
  }
  return buildSession(storedUser);
}

function persistSession(user: StoredUser): Session {
  const session = buildSession(user);
  const storedSession: StoredSession = {
    userId: user.id,
    expiresAt: Date.now() + session.expires_in * 1000,
  };
  saveStoredSession(storedSession);
  return session;
}

function notify(event: AuthChangeEvent, session: Session | null): void {
  listeners.forEach((listener) => {
    try {
      listener(event, session);
    } catch (error) {
      console.error("mock_supabase_listener_failed", error);
    }
  });
}

function createAuthError(message: string, status = 400): AuthError {
  return {
    name: "AuthApiError",
    message,
    status,
  } as AuthError;
}

export function createMockSupabaseClient(): SupabaseClient {
  return {
    auth: {
      async getSession() {
        const session = getActiveSession();
        return { data: { session }, error: null };
      },
      async signInWithPassword({ email, password }: { email: string; password: string }) {
        const users = loadUsers();
        const storedUser = Object.values(users).find((user) => user.email === email);
        if (!storedUser || storedUser.password !== password) {
          return {
            data: { user: null, session: null },
            error: createAuthError("Invalid email or password", 400),
          };
        }
        const session = persistSession(storedUser);
        notify("SIGNED_IN", session);
        return { data: { user: session.user, session }, error: null };
      },
      async signUp({
        email,
        password,
        options,
      }: {
        email: string;
        password: string;
        options?: { data?: { full_name?: string } };
      }) {
        const users = loadUsers();
        const exists = Object.values(users).some((user) => user.email === email);
        if (exists) {
          return {
            data: { user: null, session: null },
            error: createAuthError("User already exists", 409),
          };
        }
        const now = new Date().toISOString();
        const storedUser: StoredUser = {
          id: uuid(),
          email,
          password,
          name: options?.data?.full_name ?? "",
          createdAt: now,
          updatedAt: now,
        };
        users[storedUser.id] = storedUser;
        saveUsers(users);
        const session = persistSession(storedUser);
        notify("SIGNED_IN", session);
        return { data: { user: session.user, session }, error: null };
      },
      async signOut() {
        saveStoredSession(null);
        notify("SIGNED_OUT", null);
        return { error: null };
      },
      onAuthStateChange(callback: AuthListener) {
        listeners.add(callback);
        const currentSession = getActiveSession();
        callback(currentSession ? "SIGNED_IN" : "SIGNED_OUT", currentSession);
        return {
          data: {
            subscription: {
              unsubscribe() {
                listeners.delete(callback);
              },
            },
          },
        };
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;
}
