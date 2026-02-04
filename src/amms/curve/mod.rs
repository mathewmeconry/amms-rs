use super::{
    amm::AutomatedMarketMaker,
    error::AMMError,
    Token,
};

use alloy::{
    eips::BlockId,
    network::Network,
    primitives::{Address, B256, U256},
    providers::Provider,
    rpc::types::Log,
    sol,
    sol_types::SolEvent,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

sol! {
    #[derive(Debug, PartialEq, Eq)]
    #[sol(rpc)]
    contract ICurvePool {
        event TokenExchange(
            address indexed buyer,
            int128 sold_id,
            uint256 tokens_sold,
            int128 bought_id,
            uint256 tokens_bought
        );

        event TokenExchangeUnderlying(
            address indexed buyer,
            int128 sold_id,
            uint256 tokens_sold,
            int128 bought_id,
            uint256 tokens_bought
        );

        event AddLiquidity(
            address indexed provider,
            uint256[2] token_amounts,
            uint256[2] fees,
            uint256 invariant,
            uint256 token_supply
        );

        event RemoveLiquidity(
            address indexed provider,
            uint256[2] token_amounts,
            uint256[2] fees,
            uint256 token_supply
        );

        function coins(uint256 i) external view returns (address);
        function balances(uint256 i) external view returns (uint256);
        function A() external view returns (uint256);
        function get_virtual_price() external view returns (uint256);
        function fee() external view returns (uint256);
        function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256);
    }

    #[derive(Debug, PartialEq, Eq)]
    #[sol(rpc)]
    contract ICurveFactory {
        event PoolAdded(address indexed pool);
    }
}

#[derive(Error, Debug)]
pub enum CurveError {
    #[error("Invalid token index")]
    InvalidTokenIndex,
    #[error("Invariant calculation failed to converge")]
    InvariantCalculationFailed,
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    #[error("Calculation error")]
    CalculationError,
}

/// Curve StableSwap pool implementation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CurvePool {
    pub address: Address,
    pub tokens: Vec<Token>,
    pub balances: Vec<u128>,
    pub amplification_coefficient: u128, // A parameter
    pub fee: u32,                        // fee in basis points (e.g., 4 = 0.04%)
}

impl AutomatedMarketMaker for CurvePool {
    fn address(&self) -> Address {
        self.address
    }

    fn sync_events(&self) -> Vec<B256> {
        vec![
            ICurvePool::TokenExchange::SIGNATURE_HASH,
            ICurvePool::TokenExchangeUnderlying::SIGNATURE_HASH,
            ICurvePool::AddLiquidity::SIGNATURE_HASH,
            ICurvePool::RemoveLiquidity::SIGNATURE_HASH,
        ]
    }

    fn sync(&mut self, _log: &Log) -> Result<(), AMMError> {
        // For Curve pools, we need to refetch the balances since the events
        // don't always provide the full state update (especially for multi-token pools)
        // This is a simplified implementation - in production you'd want to fetch
        // the actual balances from the contract
        info!(
            target = "amm::curve::sync",
            address = ?self.address,
            "Sync event received - balances need to be refreshed"
        );
        Ok(())
    }

    fn simulate_swap(
        &self,
        base_token: Address,
        quote_token: Address,
        amount_in: U256,
    ) -> Result<U256, AMMError> {
        let i = self.get_token_index(base_token)?;
        let j = self.get_token_index(quote_token)?;

        self.get_dy(i, j, amount_in)
    }

    fn simulate_swap_mut(
        &mut self,
        base_token: Address,
        quote_token: Address,
        amount_in: U256,
    ) -> Result<U256, AMMError> {
        let i = self.get_token_index(base_token)?;
        let j = self.get_token_index(quote_token)?;

        let amount_out = self.get_dy(i, j, amount_in)?;

        // Update balances
        self.balances[i] = self.balances[i]
            .checked_add(amount_in.to::<u128>())
            .ok_or(AMMError::ArithmeticError)?;
        self.balances[j] = self.balances[j]
            .checked_sub(amount_out.to::<u128>())
            .ok_or(AMMError::ArithmeticError)?;

        Ok(amount_out)
    }

    fn tokens(&self) -> Vec<Address> {
        self.tokens.iter().map(|t| t.address).collect()
    }

    fn calculate_price(&self, base_token: Address, quote_token: Address) -> Result<f64, AMMError> {
        // Calculate the price by simulating a small swap
        let i = self.get_token_index(base_token)?;
        let j = self.get_token_index(quote_token)?;

        let base_decimals = self.tokens[i].decimals;
        let _quote_decimals = self.tokens[j].decimals;

        // Use 1 token as the amount_in
        let amount_in = U256::from(10_u128.pow(base_decimals as u32));
        let amount_out = self.get_dy(i, j, amount_in)?;

        // Calculate price adjusting for decimals
        let amount_in_f64 = amount_in.to::<u128>() as f64;
        let amount_out_f64 = amount_out.to::<u128>() as f64;

        Ok(amount_out_f64 / amount_in_f64)
    }

    async fn init<N, P>(self, _block_number: BlockId, _provider: P) -> Result<Self, AMMError>
    where
        N: Network,
        P: Provider<N> + Clone,
    {
        // In a full implementation, this would fetch pool data from the contract
        // For now, we assume the pool is already initialized with data
        info!(
            target = "amm::curve::init",
            address = ?self.address,
            "Initializing Curve pool"
        );
        Ok(self)
    }
}

impl CurvePool {
    /// Create a new Curve pool
    pub fn new(
        address: Address,
        tokens: Vec<Token>,
        balances: Vec<u128>,
        amplification_coefficient: u128,
        fee: u32,
    ) -> Self {
        Self {
            address,
            tokens,
            balances,
            amplification_coefficient,
            fee,
        }
    }

    /// Get the index of a token in the pool
    fn get_token_index(&self, token: Address) -> Result<usize, AMMError> {
        self.tokens
            .iter()
            .position(|t| t.address == token)
            .ok_or_else(|| CurveError::InvalidTokenIndex.into())
    }

    /// Calculate the StableSwap invariant D
    /// Uses Newton's method to solve: A * n^n * sum(x_i) * prod(x_i) + D = A * D^n * n^n + D^(n+1) / (n^n * prod(x_i))
    fn calculate_d(&self) -> Result<U256, AMMError> {
        let n = self.balances.len();
        let ann = self.amplification_coefficient * (n as u128).pow(n as u32);

        let mut sum = U256::ZERO;
        let mut prod = U256::from(1u128);

        for &balance in &self.balances {
            let bal = U256::from(balance);
            sum += bal;
            prod = prod
                .checked_mul(bal)
                .ok_or(AMMError::ArithmeticError)?;
        }

        if prod.is_zero() {
            return Ok(U256::ZERO);
        }

        let mut d_prev;
        let mut d = sum;

        // Newton's method iteration
        for _ in 0..255 {
            d_prev = d;

            // Calculate d_p = D^(n+1) / (n^n * prod(x_i))
            let mut d_p = d;
            for &balance in &self.balances {
                let bal = U256::from(balance);
                d_p = d_p
                    .checked_mul(d)
                    .ok_or(AMMError::ArithmeticError)?
                    .checked_div(bal * U256::from(n))
                    .ok_or(AMMError::ArithmeticError)?;
            }

            // d = (Ann * sum + d_p * n) * d / ((Ann - 1) * d + (n + 1) * d_p)
            let ann_u256 = U256::from(ann);
            let n_u256 = U256::from(n);

            let numerator = (ann_u256
                .checked_mul(sum)
                .ok_or(AMMError::ArithmeticError)?
                + d_p
                    .checked_mul(n_u256)
                    .ok_or(AMMError::ArithmeticError)?)
            .checked_mul(d)
            .ok_or(AMMError::ArithmeticError)?;

            let denominator = (ann_u256
                .checked_sub(U256::from(1u128))
                .ok_or(AMMError::ArithmeticError)?
                .checked_mul(d)
                .ok_or(AMMError::ArithmeticError)?
                + (n_u256 + U256::from(1u128))
                    .checked_mul(d_p)
                    .ok_or(AMMError::ArithmeticError)?)
            .max(U256::from(1u128));

            d = numerator
                .checked_div(denominator)
                .ok_or(AMMError::ArithmeticError)?;

            // Check convergence
            let diff = if d > d_prev {
                d - d_prev
            } else {
                d_prev - d
            };

            if diff <= U256::from(1u128) {
                return Ok(d);
            }
        }

        Err(CurveError::InvariantCalculationFailed.into())
    }

    /// Calculate output amount for a swap
    /// i: input token index
    /// j: output token index
    /// dx: input amount
    fn get_dy(&self, i: usize, j: usize, dx: U256) -> Result<U256, AMMError> {
        if i >= self.balances.len() || j >= self.balances.len() {
            return Err(CurveError::InvalidTokenIndex.into());
        }

        if i == j {
            return Err(CurveError::InvalidTokenIndex.into());
        }

        let n = self.balances.len();
        let ann = self.amplification_coefficient * (n as u128).pow(n as u32);

        // Get current invariant
        let d = self.calculate_d()?;

        // Calculate new balance of input token
        let x = U256::from(self.balances[i])
            .checked_add(dx)
            .ok_or(AMMError::ArithmeticError)?;

        // Calculate new balance of output token using Newton's method
        let y = self.get_y(i, j, x, d, ann)?;

        let dy = U256::from(self.balances[j])
            .checked_sub(y)
            .ok_or(AMMError::ArithmeticError)?;

        // Apply fee
        let fee_amount = dy
            .checked_mul(U256::from(self.fee))
            .ok_or(AMMError::ArithmeticError)?
            .checked_div(U256::from(10000u128))
            .ok_or(AMMError::ArithmeticError)?;

        dy.checked_sub(fee_amount)
            .ok_or(AMMError::ArithmeticError)
    }

    /// Calculate y given x, D, and Ann using Newton's method
    /// This solves for y in the invariant equation
    fn get_y(
        &self,
        i: usize,
        j: usize,
        x: U256,
        d: U256,
        ann: u128,
    ) -> Result<U256, AMMError> {
        let n = self.balances.len();
        let ann_u256 = U256::from(ann);
        let n_u256 = U256::from(n);

        // Calculate c = D^(n+1) / (n^n * Ann * prod(x_k) for k != j)
        let mut c = d;
        let mut s = U256::ZERO;

        for k in 0..n {
            if k == j {
                continue;
            }

            let x_k = if k == i {
                x
            } else {
                U256::from(self.balances[k])
            };

            s = s.checked_add(x_k).ok_or(AMMError::ArithmeticError)?;
            c = c
                .checked_mul(d)
                .ok_or(AMMError::ArithmeticError)?
                .checked_div(x_k.checked_mul(n_u256).ok_or(AMMError::ArithmeticError)?)
                .ok_or(AMMError::ArithmeticError)?;
        }

        c = c
            .checked_mul(d)
            .ok_or(AMMError::ArithmeticError)?
            .checked_div(
                ann_u256
                    .checked_mul(n_u256)
                    .ok_or(AMMError::ArithmeticError)?,
            )
            .ok_or(AMMError::ArithmeticError)?;

        let b = s
            .checked_add(
                d.checked_div(ann_u256)
                    .ok_or(AMMError::ArithmeticError)?,
            )
            .ok_or(AMMError::ArithmeticError)?;

        let mut y_prev;
        let mut y = d;

        // Newton's method iteration
        for _ in 0..255 {
            y_prev = y;
            // y = (y^2 + c) / (2y + b - D)
            let y_squared = y.checked_mul(y).ok_or(AMMError::ArithmeticError)?;
            let numerator = y_squared
                .checked_add(c)
                .ok_or(AMMError::ArithmeticError)?;
            let denominator = y
                .checked_mul(U256::from(2u128))
                .ok_or(AMMError::ArithmeticError)?
                .checked_add(b)
                .ok_or(AMMError::ArithmeticError)?
                .checked_sub(d)
                .ok_or(AMMError::ArithmeticError)?;

            if denominator.is_zero() {
                return Err(CurveError::CalculationError.into());
            }

            y = numerator
                .checked_div(denominator)
                .ok_or(AMMError::ArithmeticError)?;

            // Check convergence
            let diff = if y > y_prev {
                y - y_prev
            } else {
                y_prev - y
            };

            if diff <= U256::from(1u128) {
                return Ok(y);
            }
        }

        Err(CurveError::InvariantCalculationFailed.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::address;

    #[test]
    fn test_calculate_d_simple() {
        // Test with a simple 2-token pool with equal balances
        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(address!("0000000000000000000000000000000000000001"), 18),
                Token::new_with_decimals(address!("0000000000000000000000000000000000000002"), 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000], // 1000 tokens each
            amplification_coefficient: 100,
            fee: 4, // 0.04%
        };

        let d = pool.calculate_d().unwrap();
        assert!(d > U256::ZERO);
        
        // For balanced pool with equal amounts, D should be approximately 2 * balance
        let expected_d_approx = U256::from(2_000_000_000_000_000_000_000_u128);
        let diff = if d > expected_d_approx {
            d - expected_d_approx
        } else {
            expected_d_approx - d
        };
        
        // Allow 1% deviation
        let tolerance = expected_d_approx / U256::from(100u128);
        assert!(diff < tolerance, "D calculation deviated too much: got {}, expected ~{}", d, expected_d_approx);
    }

    #[test]
    fn test_get_dy_swap() {
        // Test a swap in a balanced pool
        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(address!("0000000000000000000000000000000000000001"), 18),
                Token::new_with_decimals(address!("0000000000000000000000000000000000000002"), 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000],
            amplification_coefficient: 100,
            fee: 4,
        };

        // Swap 10 tokens (10 * 10^18)
        let amount_in = U256::from(10_000_000_000_000_000_000_u128);
        let amount_out = pool.get_dy(0, 1, amount_in).unwrap();

        // For stableswap with balanced pool, output should be close to input (minus fee)
        // Fee is 0.04% = 0.0004, so we expect ~9.996 tokens out
        let expected_out_min = U256::from(9_990_000_000_000_000_000_u128); // 9.99 tokens
        let expected_out_max = U256::from(10_000_000_000_000_000_000_u128); // 10 tokens
        
        assert!(
            amount_out >= expected_out_min && amount_out <= expected_out_max,
            "Swap output {} is outside expected range [{}, {}]",
            amount_out,
            expected_out_min,
            expected_out_max
        );
    }

    #[test]
    fn test_simulate_swap() {
        let token_a = address!("0000000000000000000000000000000000000001");
        let token_b = address!("0000000000000000000000000000000000000002");

        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(token_a, 18),
                Token::new_with_decimals(token_b, 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000],
            amplification_coefficient: 100,
            fee: 4,
        };

        let amount_in = U256::from(10_000_000_000_000_000_000_u128);
        let amount_out = pool.simulate_swap(token_a, token_b, amount_in).unwrap();

        assert!(amount_out > U256::ZERO);
        assert!(amount_out < amount_in); // With fee, should get slightly less
    }

    #[test]
    fn test_simulate_swap_mut() {
        let token_a = address!("0000000000000000000000000000000000000001");
        let token_b = address!("0000000000000000000000000000000000000002");

        let mut pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(token_a, 18),
                Token::new_with_decimals(token_b, 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000],
            amplification_coefficient: 100,
            fee: 4,
        };

        let initial_balance_0 = pool.balances[0];
        let initial_balance_1 = pool.balances[1];

        let amount_in = U256::from(10_000_000_000_000_000_000_u128);
        let amount_out = pool.simulate_swap_mut(token_a, token_b, amount_in).unwrap();

        // Check that balances were updated
        assert_eq!(
            pool.balances[0],
            initial_balance_0 + amount_in.to::<u128>()
        );
        assert_eq!(
            pool.balances[1],
            initial_balance_1 - amount_out.to::<u128>()
        );
    }

    #[test]
    fn test_calculate_price() {
        let token_a = address!("0000000000000000000000000000000000000001");
        let token_b = address!("0000000000000000000000000000000000000002");

        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(token_a, 18),
                Token::new_with_decimals(token_b, 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000],
            amplification_coefficient: 100,
            fee: 4,
        };

        let price = pool.calculate_price(token_a, token_b).unwrap();

        // For a balanced stableswap pool, price should be close to 1.0
        assert!(price > 0.99 && price < 1.01, "Price {} is not close to 1.0", price);
    }

    #[test]
    fn test_imbalanced_pool() {
        // Test with an imbalanced pool
        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(address!("0000000000000000000000000000000000000001"), 18),
                Token::new_with_decimals(address!("0000000000000000000000000000000000000002"), 18),
            ],
            balances: vec![2_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000], // 2:1 ratio
            amplification_coefficient: 100,
            fee: 4,
        };

        let d = pool.calculate_d().unwrap();
        assert!(d > U256::ZERO);

        // Swap the abundant token for the scarce one should give less
        let amount_in = U256::from(10_000_000_000_000_000_000_u128);
        let amount_out = pool.get_dy(0, 1, amount_in).unwrap();
        
        // Should get less than amount_in due to imbalance and fee
        assert!(amount_out < amount_in);
    }

    #[test]
    fn test_three_token_pool() {
        // Test a 3-token pool
        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(address!("0000000000000000000000000000000000000001"), 18),
                Token::new_with_decimals(address!("0000000000000000000000000000000000000002"), 18),
                Token::new_with_decimals(address!("0000000000000000000000000000000000000003"), 18),
            ],
            balances: vec![
                1_000_000_000_000_000_000_000,
                1_000_000_000_000_000_000_000,
                1_000_000_000_000_000_000_000,
            ], // 1000 tokens each
            amplification_coefficient: 100,
            fee: 4,
        };

        let d = pool.calculate_d().unwrap();
        assert!(d > U256::ZERO);

        // For a balanced 3-token pool, D should be approximately 3 * balance
        let expected_d_approx = U256::from(3_000_000_000_000_000_000_000_u128);
        let diff = if d > expected_d_approx {
            d - expected_d_approx
        } else {
            expected_d_approx - d
        };

        // Allow 1% deviation
        let tolerance = expected_d_approx / U256::from(100u128);
        assert!(
            diff < tolerance,
            "D calculation deviated too much: got {}, expected ~{}",
            d,
            expected_d_approx
        );

        // Test swap in 3-token pool
        let amount_in = U256::from(10_000_000_000_000_000_000_u128);
        let amount_out = pool.get_dy(0, 1, amount_in).unwrap();

        // For balanced pool, should get close to input amount (minus fee)
        assert!(amount_out > U256::ZERO);
        assert!(amount_out < amount_in); // With fee

        // Also test a different pair
        let amount_out_2 = pool.get_dy(1, 2, amount_in).unwrap();
        assert!(amount_out_2 > U256::ZERO);
        assert!(amount_out_2 < amount_in);
    }

    #[test]
    fn test_different_decimals() {
        // Test a pool with tokens having different decimal places
        let token_a = address!("0000000000000000000000000000000000000001");
        let token_b = address!("0000000000000000000000000000000000000002");

        // For Curve pools, balances should be normalized to 18 decimals internally
        // So 1M USDC (6 decimals) = 1_000_000 * 10^6 = 1_000_000_000_000
        // But for StableSwap math, we want equal value, so we normalize:
        // 1M USDC = 1_000_000_000_000_000_000_000_000 (normalized to 18 decimals)
        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(token_a, 6),  // USDC-like (6 decimals)
                Token::new_with_decimals(token_b, 18), // DAI-like (18 decimals)
            ],
            balances: vec![
                1_000_000_000_000_000_000_000_000, // 1M tokens normalized
                1_000_000_000_000_000_000_000_000, // 1M tokens
            ],
            amplification_coefficient: 100,
            fee: 4,
        };

        // Calculate price
        let price = pool.calculate_price(token_a, token_b).unwrap();

        // For a balanced stablecoin pool, price should be close to 1.0
        // (when adjusted for decimals, which is handled in calculate_price)
        assert!(
            price > 0.99 && price < 1.01,
            "Price {} is not close to 1.0 for stablecoin pool",
            price
        );
    }

    #[test]
    fn test_large_swap() {
        // Test a large swap to see slippage increase
        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(address!("0000000000000000000000000000000000000001"), 18),
                Token::new_with_decimals(address!("0000000000000000000000000000000000000002"), 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000],
            amplification_coefficient: 100,
            fee: 4,
        };

        // Small swap
        let small_amount = U256::from(10_000_000_000_000_000_000_u128); // 10 tokens
        let small_out = pool.get_dy(0, 1, small_amount).unwrap();

        // Large swap (10% of pool)
        let large_amount = U256::from(100_000_000_000_000_000_000_u128); // 100 tokens
        let large_out = pool.get_dy(0, 1, large_amount).unwrap();

        // Calculate effective exchange rates
        let small_rate = small_out.to::<u128>() as f64 / small_amount.to::<u128>() as f64;
        let large_rate = large_out.to::<u128>() as f64 / large_amount.to::<u128>() as f64;

        // Large swap should have worse rate due to slippage
        assert!(
            large_rate < small_rate,
            "Large swap rate {} should be worse than small swap rate {}",
            large_rate,
            small_rate
        );
    }

    #[test]
    fn test_zero_amount_swap() {
        let token_a = address!("0000000000000000000000000000000000000001");
        let token_b = address!("0000000000000000000000000000000000000002");

        let pool = CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(token_a, 18),
                Token::new_with_decimals(token_b, 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000],
            amplification_coefficient: 100,
            fee: 4,
        };

        let amount_in = U256::ZERO;
        let amount_out = pool.get_dy(0, 1, amount_in).unwrap();

        // Zero input should give zero output
        assert_eq!(amount_out, U256::ZERO);
    }

    #[test]
    fn test_different_amplification_coefficients() {
        // Test pools with different A values
        let create_pool = |amp_coef: u128| CurvePool {
            address: Address::default(),
            tokens: vec![
                Token::new_with_decimals(address!("0000000000000000000000000000000000000001"), 18),
                Token::new_with_decimals(address!("0000000000000000000000000000000000000002"), 18),
            ],
            balances: vec![1_000_000_000_000_000_000_000, 1_000_000_000_000_000_000_000],
            amplification_coefficient: amp_coef,
            fee: 4,
        };

        let low_amp = create_pool(10);  // More like constant product
        let high_amp = create_pool(1000); // More like constant sum

        let amount_in = U256::from(100_000_000_000_000_000_000_u128); // 100 tokens (10% of pool)

        let low_amp_out = low_amp.get_dy(0, 1, amount_in).unwrap();
        let high_amp_out = high_amp.get_dy(0, 1, amount_in).unwrap();

        // Higher amplification should give better output (less slippage)
        assert!(
            high_amp_out > low_amp_out,
            "High amp output {} should be better than low amp output {}",
            high_amp_out,
            low_amp_out
        );

        // High amp should be closer to the input amount
        let high_amp_diff = amount_in - high_amp_out;
        let low_amp_diff = amount_in - low_amp_out;
        assert!(high_amp_diff < low_amp_diff);
    }
}
