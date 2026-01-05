"use client";

import React, { useEffect, useState } from "react";
import Navbar from "@/components/navbar/page";
import { supabase } from "@/lib/supabaseClient";
const formatName = (user) => {
  if (user?.user_metadata?.name) {
    return user.user_metadata.name;
  }
  if (user?.email) {
    return user.email.split("@")[0];
  }
  return "User";
};

const Dashboard = () => {
  const [name, setName] = useState("");

  useEffect(() => {
    const cachedName = localStorage.getItem("user_name");
    if (cachedName) {
      setName(cachedName);
    }

    const fetchUser = async () => {
      const { data } = await supabase.auth.getUser();
      if (!data?.user) return;

      const user = data.user;

      const resolvedName = formatName(user);

      setName(resolvedName);
      localStorage.setItem("user_name", resolvedName);
    };

    fetchUser();
  }, []);

  return (
    <>
      <Navbar />
      <h2>
        Welcome  {name}
      </h2>
    </>
  );
};

export default Dashboard;
