"use client";
import React from 'react';
import styles from './styles.module.css';
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabaseClient";
const Navbar = () => {
  const router = useRouter();
  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      console.error("Error logging out:", error);
    } else {
      router.push("/auth");
    }
  };
  return (
    <div className={styles.nav}>
      <div className={styles.logo}>
        <h1>Dashboard</h1>
      </div>
      <div className={styles.getStarted} onClick={handleLogout}>
            logout
      </div>
    </div>
  );
};

export default Navbar;
