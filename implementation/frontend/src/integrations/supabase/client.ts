import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";
import { createMockSupabaseClient } from "./mockClient";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

let supabase: SupabaseClient;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn(
    "Supabase credentials are not configured. Falling back to mock authentication. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to use Supabase.",
  );
  supabase = createMockSupabaseClient();
} else {
  supabase = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: false,
      storageKey: "neuraforge-auth",
    },
  });
}

export { supabase };
