/*
Write a smart contract on a test network, for Bank account of a customer for following
operations:
 Deposit money
 Withdraw Money
 Show balance
*/



// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BankAccount {
    mapping(address => uint256) private balances;
    constructor() payable {
        // Optionally, initialize the contract's balance if needed
        if (msg.value > 0) {
            balances[msg.sender] = msg.value;
        }
    }

    function deposit() public payable {
        require(msg.value > 0, "You must deposit a positive amount.");
        balances[msg.sender] += msg.value; 
    }
    function withdraw(uint256 amount) public {
        require(amount > 0, "You must withdraw a positive amount.");
        require(balances[msg.sender] >= amount, "Insufficient balance.");

        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount); 
    }
    function getBalance() public view returns (uint256) {
        return balances[msg.sender]; 
    }
}
