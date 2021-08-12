// Fill out your copyright notice in the Description page of Project Settings.


#include "PlayerCharacter.h"

// Sets default values
APlayerCharacter::APlayerCharacter()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void APlayerCharacter::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void APlayerCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	FVector userLoc = GetWorld()->GetFirstPlayerController()->GetPawn()->GetActorLocation();

	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(1,2.f,FColor::Blue, FString::Printf(TEXT("Location: %s"),*userLoc.ToString()));
	}

	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(1, 2.f, FColor::Blue, FString("Hello"));
	}
}

// Called to bind functionality to input
void APlayerCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}
