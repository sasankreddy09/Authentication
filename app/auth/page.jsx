'use client';

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabaseClient";
import styles from "./styles.module.css";
function LoginComponent() {
  const router = useRouter();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [isLoginMethod, setIsLoginMethod] = useState(true);
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      if (data.session) {
        router.push("/dashboard");
      }
      else{
        router.push("/auth");
      }
    });
  }, [router]);

  useEffect(() => {
    setMessage("");
    setIsError(false);
  }, [isLoginMethod]);
  
  const validateFields = () => {
    if (!isLoginMethod) {
      if (!username || !name) {
        setIsError(true);
        setMessage("All fields are required for signup.");
        return false;
      }
    }
  if (!email || !password) {
    setIsError(true);
    setMessage("Email and password are required.");
    return false;
  }


  return true;
  };
const handleRegister = async () => {
  setIsError(false);
  setMessage("");

  if (!validateFields()) return;

    setLoading(true);


    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    });

    if (error) {
      setIsError(true);
      if (error.message.includes("username")) {
      setMessage("Username already exists.");
    } else{
      setMessage(error.message);
    }
      setLoading(false);
      return;
    }

    const { error: profileError } = await supabase
      .from("Users")
      .insert({
        id: data.user.id,
        username: username,
        full_name: name,
        email: email,
      });

    setLoading(false);

    if (profileError) {
      console.log(profileError)
      if (profileError.message.includes("Users_id_key")) {
      setMessage("Username already exists.");
    }else{ 
    setMessage(profileError.message);
  }
      setIsError(true);
    } else {
      setIsError(false);
      setMessage("Signup successful! Please login");
    }
  };
  const handleLogin = async () => {
    setIsError(false);
    setMessage("");

    if (!email || !password) {
      setIsError(true);
      setMessage("Email and password are required.");
      return;
    }

    setLoading(true);

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    setLoading(false);

    if (error) {
      setIsError(true);
      setMessage(error.message);
    } else {
      router.push("/dashboard");
    }
  };



  return (
    <div className={styles.container}>
        <div className={styles.cardContainer}>
            <div className={styles.cardContainer_left}>
                <div style={{ marginTop: "1rem" }}>
                  <h2>{isLoginMethod ? "Sign In" : "Sign Up"}</h2>

                  {message && (
                    <p
                      style={{
                        marginTop: "1rem",
                        color: isError ? "red" : "green",
                        fontSize: "0.9rem",
                      }}
                    >
                      {message}
                    </p>
                  )}
                </div>
                <div className={styles.inputElements}>
                    <div className={styles.inputRow}>
                    <input onChange={(e)=>setEmail(e.target.value)} className={styles.inputs} type="email" value={email} placeholder="email" />
                    <input onChange={(e)=>setPassword(e.target.value)} className={styles.inputs} type="password" value={password} placeholder="Password" />
                    </div>
                    <div className={styles.inputRow}>
                    {!isLoginMethod && (
                        <div> <input onChange={(e)=>setUsername(e.target.value)} className={styles.inputs} value={username} type="text" placeholder="username" />
                        <input onChange={(e)=>setName(e.target.value)} className={styles.inputs} value={name} type="text" placeholder="Full Name"/></div>
                    )}
                    </div>
                    <div className={styles.inputRow}>
                        <button
                            className={styles.button}
                            disabled={loading}
                            onClick={() => {
                              if (isLoginMethod) {
                                handleLogin();
                              } else {
                                handleRegister();
                              }
                            }}
                          >
                            {loading ? "Please wait..." : isLoginMethod ? "Sign In" : "Sign Up"}
                          </button>
                    </div>
                </div>
            </div>
            <div className={styles.cardContainer_right}>
                {isLoginMethod?(
                    <div>
                        <h3>New to Website?</h3>
                        <button onClick={()=>setIsLoginMethod(false)} className={styles.buttons}>Sign Up</button>
                    </div>
                ):(
                    <div>
                       <h3>Already have an account?</h3>{" "}
                        <button onClick={()=>setIsLoginMethod(true)} className={styles.buttons}>Sign In</button>
                    </div>
                )}
            </div>
        </div>
        </div>
  );
}

export default LoginComponent;
