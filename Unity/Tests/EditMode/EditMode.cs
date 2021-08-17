using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
/**
 * All tests within this class are integration tests to ensure GRPC communication is working correctly
 * All coordinates are parameterised to be between 0 and 1
 */
public class EditMode
{
    /**
     * Checks that image sent back match the image created by the server with the same coordinates
     * Test 1 - Using FashionMNIST
     */
    [Test]
    public async void GRPCTestEqual()
    {
        GRPCClient grpcClient = new GRPCClient(); 
        Vector3 coords = new Vector3(0.5f, 0.6f, 0.7f);
        //fetches image based on coords from server
        byte[] test = await grpcClient.HandleCoords(coords);
        //fetches pre saved image
        byte[] actual = File.ReadAllBytes("./Assets/Tests/EditMode/17082021172845.png");
        Assert.AreEqual(test, actual);
    }
    
    /**
     * This test checks that that if different coordinates are sent that we do not get the same image as tested above
     * All unique coordinates should return different images
     * Test 1
     */
    [Test]
    public async void GRPCTestNotEqual()
    {
        GRPCClient grpcClient = new GRPCClient(); 
        Vector3 coords = new Vector3(5.8f, 0.2f, 0.9f);
        //fetches image based on coords from server
        byte[] test = await grpcClient.HandleCoords(coords);
        //fetches pre saved image
        byte[] actual = File.ReadAllBytes("./Assets/Tests/EditMode/17082021172845.png");
        Assert.AreNotEqual(test, actual);
    }
    
    /**
     * Checks that image sent back match the image created by the server with the same coordinates
     * Test 2 - Using FashionMNIST
     */
    [Test]
    public async void GRPCTestEqual2()
    {
        GRPCClient grpcClient = new GRPCClient(); 
        Vector3 coords = new Vector3(0.8f, 0.1f, 0.2f);
        //fetches image based on coords from server
        byte[] test = await grpcClient.HandleCoords(coords);
        //fetches pre saved image
        byte[] actual = File.ReadAllBytes("./Assets/Tests/EditMode/17082021195359.png");
        Assert.AreEqual(test, actual);
    }
    
    /**
     * This test checks that that if different coordinates are sent that we do not get the same image as tested above
     * All unique coordinates should return different images
     * Test 2
     */
    [Test]
    public async void GRPCTestNotEqual2()
    {
        GRPCClient grpcClient = new GRPCClient(); 
        Vector3 coords = new Vector3(0.8f, 0.3f, 0.4f);
        //fetches image based on coords from server
        byte[] test = await grpcClient.HandleCoords(coords);
        //fetches pre saved image
        byte[] actual = File.ReadAllBytes("./Assets/Tests/EditMode/17082021195359.png");
        Assert.AreNotEqual(test, actual);
    }
}
