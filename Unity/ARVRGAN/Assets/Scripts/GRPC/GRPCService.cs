using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class GRPCService : MonoBehaviour
{
    //private Image;
    private GRPCClient client;

    private GenerativeModel pythonClient;
    //public Material imgPlane;
    public GameObject plane;
    public GameObject camera;

    private void Start()
    {
        //plane.GetComponent<Renderer>().material.color = Color.white;
        //pythonClient = new GenerativeModel();
        //pythonClient.FetchImagePython();
    }

    // Start is called before the first frame update
    void LateUpdate()
    {
        plane.GetComponent<Renderer>().material.color = Color.cyan;
        client = new GRPCClient();
        client.HandleCoords(plane, camera);
        //GenerativeModel gm = new GenerativeModel();
        //gm.FetchImagePython();
    } }
